#!/usr/bin/env python
import optparse, sys, math, models_new
from collections import namedtuple
from numpy import matrix, array


optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="dev/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="large/phrase-table/test-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-w", "--weight-file", dest="weights", default=None, help="Weight filename, or - for stdin (default=use uniform weights)")
#optparser.add_option("-i", "--input", dest="input", default="toy/train.cn", help="File containing sentences to translate (default=data/input)")
#optparser.add_option("-t", "--translation-model", dest="tm", default="toy/phrase-table/phrase_table.out", help="File containing translation model (default=data/tm)")
#optparser.add_option("-l", "--language-model", dest="lm", default="lm/en.tiny.3g.arpa", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-d", "--distortion-limit", dest="d", default=0, type="int", help="Distortion limit (default=0)")
optparser.add_option("-f", "--future-cost", dest="future_cost", action="store_true", default=False,  help="Compute future cost estimation (default=false)")
optparser.add_option("-b", "--nbest", dest="nbest", default=1, type="int", help="Number of sentences (in nbest) (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]


# language model weight - phrase table weights - distortion penalty weight
if opts.weights is not None:
	weights_file = sys.stdin if opts.weights is "-" else open(opts.weights)
	weights = [float(line.strip()) for line in weights_file]
	weights = map(lambda x: 1.0 if math.isnan(x) or x == float("-inf") or x == float("inf") or x == 0.0 else x, weights)
	weights.append(0)
	weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0] if len(weights) == 0 else weights
else:
	weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0]

tm = models_new.TM(opts.tm, opts.k, weights[1:5])
lm = models_new.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
	if (word,) not in tm:
		tm[(word,)] = [models_new.phrase(word, matrix([0, 0, 0, 0]), 0)]

# Negative infinity value
neginf = -999999

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for (num,f) in enumerate(french):	
	sys.stderr.write("\r%d/%d\t" % (num+1,len(french)))
	
	# Compute Future Cost Estimation
	if opts.future_cost:
		FCost = [[neginf for _ in range(0, len(f))] for _ in range(0, len(f))]
		
		# Find and compute best translation options for all input spans
		for i in range(0, len(f)):
			for j in range(i, len(f)):
				if f[i:j+1] in tm:							
					best_phrase = max(tm[f[i:j+1]],key=lambda h: h.score)
					logprob = best_phrase.score
					
					for word in best_phrase.english.split():
						(lm_state, word_logprob) = lm.score((), word)
						logprob += word_logprob
				
					FCost[i][j] = logprob
		
		# Update future cost estimation for non-computed values
		for i in range(len(f)-2, -1, -1):
			for j in range(i+1, len(f)):				
				bestscore = max(FCost[i][j], FCost[i][j-1] + FCost[j][j])
				k = j-1
				while k > i:
					bestscore = max(bestscore, FCost[i][k-1] + FCost[k][j])
					k -= 1					
				FCost[i][j] = bestscore
	
	
	hypothesis = namedtuple("hypothesis", "score, lm_state, predecessor, phrase, bit, end_idx, future_cost, features")
	initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, tuple(0 for _ in f), 0, 0, [0, 0, 0, 0, 0])

	stacks = [{} for _ in f] + [{}]
	
	# If we discard the distortion penalty, then we don't need the end_idx value to compare hypotheses
	stacks[0][(initial_hypothesis.lm_state, initial_hypothesis.bit, initial_hypothesis.end_idx)] = initial_hypothesis	
  
	for (translated, stack) in enumerate(stacks[:-1]):
		for h in sorted(stack.itervalues(),key=lambda h: -(h.score + h.future_cost))[:opts.s]: # prune
			first_uncovered_pos = h.bit.index(0)
			for i in xrange(first_uncovered_pos, min(first_uncovered_pos+opts.d+1,len(f))):
				if h.bit[i] == 1: continue
				for j in xrange(i+1,len(f)+1):
					if sum(h.bit[i:j]) > 0: continue
					if f[i:j] in tm:
						for phrase in tm[f[i:j]]:
							# feature vector
							features = list(h.features)
							
							# add log translation model probability
							score = h.score + matrix(weights[1:5])*phrase.scores.T			
							features[1:5] = list(array((matrix(features[1:5]) + phrase.scores)).reshape(-1,))
	
							# add log language model probability
							lm_state = h.lm_state
							for word in phrase.english.split():
								(lm_state, word_logprob) = lm.score(lm_state, word)
								features[0] += word_logprob
								score += weights[0]*word_logprob
							if j == len(f):
								score += weights[0]*lm.end(lm_state)
								features[0] += lm.end(lm_state) 
							
							# add distortion penalty
							score += weights[5]*abs(h.end_idx - i)
							
							# Update bit string
							bit = tuple(1 if x in range(i,j) else h.bit[x] for x in range(0, len(f)))
							
							# Add Future Cost Estimation
							future_cost = 0
							if opts.future_cost:
								s = 0;
								while s < len(f):
									if(bit[s] == 0):
										e = s+1
										while e < len(f) and bit[e] == 0:
											e += 1
										future_cost += FCost[s][e-1]
										s = e+1
									else: s+=1
							
							# Create new hypothesis
							new_hypothesis = hypothesis(score, lm_state, h, phrase, bit, j, future_cost,features)
							if (lm_state, bit, j) not in stacks[sum(bit)] or stacks[sum(bit)][(lm_state, bit, j)].score < score: # second case is recombination
								stacks[sum(bit)][(lm_state, bit, j)] = new_hypothesis 

	def extract_english(h): 
		return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
								
	if opts.nbest > 1:
		for (n,winner) in enumerate(sorted(stacks[-1].itervalues(),key=lambda h: -h.score)):
			if n == opts.nbest: break;			
			translated_sentence = extract_english(winner)
			print "{0} ||| {1} |||".format(num, translated_sentence),
			print " ".join(str(feat) for feat in winner.features),
			print len(translated_sentence.split()),
			print len(set(translated_sentence.split()).intersection(f))			
	else:
		winner = max(stacks[-1].itervalues(), key=lambda h: h.score)  
		# Print translated output
		print extract_english(winner)
	
	if opts.verbose:
		def extract_tm_logprob(h):
			return 0.0 if h.predecessor is None else h.phrase.score + extract_tm_logprob(h.predecessor)
		tm_logprob = extract_tm_logprob(winner)
		sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
			(winner.score - tm_logprob, tm_logprob, winner.score))

sys.stderr.write("\nFinished. Writing output file...\n")
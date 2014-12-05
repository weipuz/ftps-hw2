#!/usr/bin/env python
import optparse
import sys
import decoder_models_4feat as models
import heapq
from collections import namedtuple
#some utilitiy function

def bitmap(sequence):
  """ Generate a coverage bitmap for a sequence of indexes """
  return reduce(lambda x,y: x|y, map(lambda i: long('1'+'0'*i,2), sequence), 0)

def bitmap2str(b, n, on='o', off='.'):
  """ Generate a length-n string representation of bitmap b """
  return '' if n==0 else (on if b&1==1 else off) + bitmap2str(b>>1, n-1, on, off)

def onbits(b):
  """ Count number of on bits in a bitmap """
  return 0 if b==0 else (1 if b&1==1 else 0) + onbits(b>>1)

def prefix1bits(b):
  """ Count number of bits encountered before first 0 """
  return 0 if b&1==0 else 1+prefix1bits(b>>1)

def last1bit(b):
  """ Return index of highest order bit that is on """
  return 0 if b==0 else 1+last1bit(b>>1)



optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="dev/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="large/phrase-table/dev-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="lm/en.gigaword.3g.filtered.dev_test.arpa.gz", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-f", "--first_num_sentences", dest="first_num_sents", default=0, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=20, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=100, type="int", help="Maximum stack size (default=5)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-e", "--eta", dest="eta", default=float(-5), type= "float") 
optparser.add_option("-d", "--distortionLimit", dest = "d", type ="int",default = 10)
optparser.add_option("-w", "--weight-file", dest="weights", default=None, help="Weight filename, or - for stdin (default=use uniform weights)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
# new tm model read phrase table and present four prob:  p(e|f), p(f|e), two lexically- weighted phrase  lex(e|f) and lex(f|e)

lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[opts.first_num_sents:(opts.first_num_sents+opts.num_sents)]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, [0.0,0.0,0.0,0.0])]
w = [0.2,-0.2,0.2,0.2,0.2,0.2]
#w = [0.5254884,-0.9,3.32859,0.351289,-0.266778,2.741809]
#w = None
if opts.weights is not None:
  weights_file = open(opts.weights)
  w = [float(line.strip()) for line in weights_file]
sys.stderr.write("Decoding %s...\n" % (opts.input,))


for idx, f in enumerate(french):
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  sys.stderr.write("Decoding %s...\n" % (str(idx),))
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, coverageVec, fpos, slogprob")
  initial_bitmap = bitmap([])
  slogprob = namedtuple("slogprob", "lms, rs, pfe, lfe, pef, lef")
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, initial_bitmap, 0, slogprob(0.0,0.0,0.0,0.0,0.0,0.0))
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin(),initial_bitmap,0] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]):
    #for h in sorted(stack.itervalues(),key=lambda h: -h.penalLogprob)[:opts.s]: # prune, penalize distortion 
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
	  firstOpen = prefix1bits(h.coverageVec)
	  for k in xrange(firstOpen, min(firstOpen+1+opts.d, len(f)+1)):
		for j in xrange(k+1,len(f)+1):
		  temp_ph = bitmap(range(k,j))
		  if (temp_ph & h.coverageVec == 0 ) and f[k:j] in tm:
			for phrase in tm[f[k:j]]:
			  #reorder the phrase logprob into the same order with the weight
			  phraselogprob=[phrase.logprob[1],phrase.logprob[3],phrase.logprob[0],phrase.logprob[2]]
			  logprob_weighted = sum([x*y for x,y in zip(w[2:6], phraselogprob)])
			  logprob = h.logprob + logprob_weighted 
			  lm_state = h.lm_state
			  #namedtuple cant be replaced, use new varible to store the value and creat new nametuple at the end
			  newpfe = h.slogprob.pfe + phrase.logprob[1]
			  newlfe = h.slogprob.lfe + phrase.logprob[3]
			  newpef = h.slogprob.pef + phrase.logprob[0]
			  newlef = h.slogprob.lef + phrase.logprob[2]
			  new_coverage= temp_ph | h.coverageVec
			  fpos = j
			  covered = onbits(new_coverage)
			  newlms = h.slogprob.lms
			  newrs = h.slogprob.rs
			  for word in phrase.english.split():
				(lm_state, word_logprob) = lm.score(lm_state, word)
				logprob += w[0]*word_logprob
				newlms += word_logprob
			  logprob += w[0]*lm.end(lm_state) if covered == len(f) else 0.0            
			  newlms += lm.end(lm_state) if covered == len(f) else 0.0            
			  newrs += abs(k-h.fpos-1)
			  logprob += w[1]*abs(k-h.fpos-1)
			  new_slogprob=slogprob(newlms,newrs, newpfe, newlfe, newpef, newlef)
			  new_hypothesis = hypothesis(logprob, lm_state, h, phrase, new_coverage, fpos, new_slogprob)
			  if (lm_state, new_coverage, fpos) not in stacks[covered] or stacks[covered][lm_state, new_coverage, fpos].logprob < logprob: # second case is recombination
				stacks[covered][lm_state, new_coverage, fpos] = new_hypothesis 
  winners = heapq.nlargest(100,stacks[-1].itervalues(), key=lambda h: h.logprob)
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  
  for winner in winners:
	print str(opts.first_num_sents+idx) + " ||| " + extract_english(winner) + " ||| "  + " ".join([str(i) for i in winner.slogprob])
  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

#!/usr/bin/env python
import optparse
import sys
import decoder_models as models
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
optparser.add_option("-i", "--input", dest="input", default="test/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="toy/phrase-table/phrase_table.out", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="lm/en.tiny.3g.arpa", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=5, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=5, type="int", help="Maximum stack size (default=5)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-e", "--eta", dest="eta", default=float(-5), type= "float") 
optparser.add_option("-d", "--distortionLimit", dest = "d", type ="int",default = 10)
opts = optparser.parse_args()[0]

tm1 = models.TM1(opts.tm, opts.k)
tm2 = models.TM2(opts.tm, opts.k)
tm3 = models.TM3(opts.tm, opts.k)
tm4 = models.TM4(opts.tm, opts.k)

lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm1:
    tm1[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, coverageVec, fpos, penalLogprob")
  initial_bitmap = bitmap([])
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, initial_bitmap, 0, 0.0)
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
			  logprob = h.logprob + phrase.logprob
			  lm_state = h.lm_state
			  new_coverage= temp_ph | h.coverageVec
			  fpos = j
			  covered = onbits(new_coverage)
			  for word in phrase.english.split():
				(lm_state, word_logprob) = lm.score(lm_state, word)
				logprob += word_logprob
			  logprob += lm.end(lm_state) if covered == len(f) else 0.0            
			  penalLogprob = logprob + opts.eta * abs(h.fpos+1-k)
			  new_hypothesis = hypothesis(logprob, lm_state, h, phrase, new_coverage, fpos, penalLogprob)
			  if (lm_state, new_coverage, fpos) not in stacks[covered] or stacks[covered][lm_state, new_coverage, fpos].logprob < logprob: # second case is recombination
				stacks[covered][lm_state, new_coverage, fpos] = new_hypothesis 
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

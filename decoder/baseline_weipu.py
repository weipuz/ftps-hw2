#!/usr/bin/env python
import optparse
import sys
import models, logging
#import score-decoder.py
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=10, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

#logging.basicConfig(filename="debug.log", filemode='w', level=logging.INFO)
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
d = 10
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

def logadd10(x,y):
  """ Addition in logspace (base 10): if x=log(a) and y=log(b), returns log(a+b) """
  return x + math.log10(1 + pow(10,y-x))








# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, bitcover")
  b= bitmap([])
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, b)
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin(),b] = initial_hypothesis
  stack_count = 0
  for i, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
	  firstopen = prefix1bits(h.bitcover)
	  for k in xrange(firstopen,min(firstopen+1+d,len(f)+1)):
		  for j in xrange(k+1,len(f)+1):
		    if f[k:j] in tm:
				temp_bitmap = bitmap(range(k, j))
				if temp_bitmap & h.bitcover == 0:	
				    new_b = temp_bitmap | h.bitcover
				    for phrase in tm[f[k:j]]:
					    logprob = h.logprob + phrase.logprob
					    lm_state = h.lm_state
					    stack_count=onbits(new_b)
					    for word in phrase.english.split():
							(lm_state, word_logprob) = lm.score(lm_state, word)
							logprob += word_logprob
					    logprob += lm.end(lm_state) if stack_count == len(f) else 0.0
					    new_hypothesis = hypothesis(logprob, lm_state, h, phrase, new_b)
					    
						#stack_count=i+1
					    if (lm_state,new_b) not in stacks[stack_count] or stacks[stack_count][lm_state,new_b].logprob <logprob: # second case is recombination
							stacks[stack_count][lm_state,new_b] = new_hypothesis 
    #logging.info(stacks[i+1])			  
  # final_index = 0
  # for index, stack in enumerate(stacks):
    # if len(stack) > 0:
		# final_index = index
  #logging.info(stacks[final_index])
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  #logging.info(winner)
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

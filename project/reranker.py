#!/usr/bin/env python
import optparse, sys, os, bleu, random
from collections import namedtuple
from math import fabs
from numpy import matrix

translation_candidate = namedtuple("candidate", "sentence, features, bleu_score")
optparser = optparse.OptionParser()
optparser.add_option("-r", "--reference", dest="reference", default=os.path.join("dev", "all.cn-en.en0"), help="English reference sentences")
optparser.add_option("-s", "--source", dest="source", default=os.path.join("dev", "all.cn-en.cn"), help="Source sentences file")
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("dev2.nbest"), help="N-best file")
optparser.add_option("-w", "--weight-file", dest="weights", default=None, help="Weight filename, or - for stdin (default=use uniform weights)")
(opts, _) = optparser.parse_args()

ref = [line.strip().split() for line in open(opts.reference)]
multiple_references = False
if multiple_references:
	ref = zip(ref, [line.strip().split() for line in open(os.path.join("dev", "all.cn-en.en1"))], [line.strip().split() for line in open(os.path.join("dev", "all.cn-en.en2"))],[line.strip().split() for line in open(os.path.join("dev", "all.cn-en.en3"))])


src = [line.strip().split() for line in open(opts.source)]
src = src[:50]
# Parameters
tau = 4000 #samples generated from n-best list per input sentence (set to 5000)
alpha = 0.08 #sampler acceptance cutoff (set to 0.1)
xi = 100 #training data generated from the samples tau (set to 100)
eta = 0.1 #perceptron learning rate (set to 0.1)
epochs = 4 #number of epochs for perceptron training (set to 5)
rand_seed = 30 # random seed #30

# Variables
nbests = []
w = []
if opts.weights is not None:
  weights_file = open(opts.weights)
  w = matrix([float(line.strip()) for line in weights_file])

sys.stderr.write("Computing smoothed bleu score for candidates")
for n, line in enumerate(open(opts.nbest)):
	
	if n % 2000 == 0:
		sys.stderr.write(".")

	(i, sentence, features) = line.strip().split("|||")
	(i, sentence) = (int(i), sentence.strip())
	
	features = [float(h) for h in features.strip().split()]
	
	if len(w) == 0:
		w = matrix([1.0/len(features) for _ in xrange(len(features))])

	features = matrix(features).T
	while len(nbests) <= i:
		nbests.append([])
	if multiple_references: 
		scores = tuple(bleu.bleu_stats2(sentence.split(), ref[i]))
	else:
		scores = tuple(bleu.bleu_stats(sentence.split(), ref[i]))
	bleu_score = bleu.smoothed_bleu(scores)
	nbests[i].append(translation_candidate(sentence, features, bleu_score))

sys.stderr.write("\nTraining...")	
avg_w = w.copy()
z = 1
random.seed(rand_seed)
for i in xrange(epochs):
	for nbest in nbests:
		# Get samples
		samples = []
		len_nbest = len(nbest)
		for k in xrange(tau):		
			# Pick randomly two candidates from nbest
			s1 = random.choice(nbest)
			s2 = random.choice(nbest)
			
			if fabs(s1.bleu_score - s2.bleu_score) > alpha:
				if s1.bleu_score > s2.bleu_score:
					samples.append((s1,s2))
				else:
					samples.append((s2,s1))
			else:
				#k -=1
				continue
				
		# Sort samples and get top xi samples
		sorted_samples = sorted(samples, key=lambda s: -fabs(s[0].bleu_score - s[1].bleu_score))[:xi]
		
		# Train from sorted samples
		for sample in sorted_samples:
			if w * sample[0].features <= w * sample[1].features:				
				w += eta * (sample[0].features - sample[1].features).T
			avg_w += w
			z += 1

avg_w = (1.0/z) * avg_w
print "\n".join([str(weight) for weight in avg_w.getA1()])
sys.stderr.write("\nCompleted")	
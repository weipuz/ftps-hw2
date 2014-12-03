#!/usr/bin/env python
import optparse
import sys, logging, random, math
import bleu
from collections import namedtuple

translation_candidate = namedtuple("candidate", "sentence, features, stats, smoothed_bleu")
optparser = optparse.OptionParser()
optparser.add_option("-r", "--reference", dest="reference", default="data/train.en", help="English reference sentences")
optparser.add_option("-n", "--nbest", dest="nbest", default="data/train.nbest", help="N-best lists")
# tau: samples generated from n-best list per input sentence (set to 5000)
optparser.add_option("-t", "--tau", dest="tau", default=5000, help="tau")
# alpha: sampler acceptance cutoff (set to 0.1)
optparser.add_option("-a", "--alpha", dest="alpha", default=0.1, help="alpha")
# xi: training data generated from the samples tau (set to 100)
optparser.add_option("-x", "--xi", dest="xi", default=100, help="xi")
# eta: perceptron learning rate (set to 0.1)
optparser.add_option("-z", "--eta", dest="eta", default=0.1, help="eta")
# epochs: number of epochs for perceptron training (set to 5)
optparser.add_option("-e", "--epochs", dest="epochs", default=5, help="epochs")
(opts,_) = optparser.parse_args()

logging.basicConfig(filename="info.log", filemode='w', level=logging.INFO)

if __name__ == '__main__':
	ref = [line.strip().split() for line in open(opts.reference)]
	nbests = []
	# for each sentence i:
	for n, line in enumerate(open(opts.nbest)):
		# collect all the n-best outputs for i
		(i, sentence, features) = line.strip().split("|||")
		(i, sentence, features) = (int(i), sentence.strip(), [float(j) for j in features.split()])
		if len(ref) <= i:
			break
		while len(nbests) <= i:
			nbests.append([])
		# for each candidate c in the n-best list:
			# compute the bleu score b (using bleu.py) for c
		stats = tuple(bleu.bleu_stats(sentence.split(), ref[i]))
		smoothed_bleu = bleu.smoothed_bleu(stats)
		# append (c,b) to nbests[i]
		nbests[i].append(translation_candidate(sentence, features, stats, smoothed_bleu))
		if n % 2000 == 0:
			sys.stderr.write(".")

	# for i = 1 to epochs:
	for i in xrange(opts.epochs):
		sys.stderr.write("epoch %d\n" % i)
		(observations, errors) = (0, 0.0)
		# for nbest in nbests:
		for nbest in nbests:
			# get_sample():
			def get_sample():
				# initialize sample to empty list
				sample = []
				# loop tau times:
				for _ in xrange(opts.tau):
					# randomly choose two items from nbest list, s1 and s2:
					c1 = random.choice(nbest)
					c2 = random.choice(nbest)
					# if fabs(s1.smoothed_bleu - s2.smoothed_bleu) > alpha:
					if c1 != c2 and math.fabs(c1.smoothed_bleu - c2.smoothed_bleu) > opts.alpha:
						# if s1.smoothed_bleu > s2.smoothed_bleu:
						if c1.smoothed_bleu > c2.smoothed_bleu:
							# sample += (s1, s2)
							sample.append([c1, c2])
						# else:
						else:
							# sample += (s2, s1)
							sample.append([c2, c1])
					# else:
					else:
						# continue
						continue
				# return sample
				return sample
			v = get_sample()
			# sort the tau samples from get_sample() using s1.smoothed_bleu - s2.smoothed_bleu
			sorted_v = sorted(v, key=lambda (c1, c2): c1.smoothed_bleu - c2.smoothed_bleu)
			# keep the top xi (s1, s2) values from the sorted list of samples
			sorted_v = sorted_v[: opts.xi]
			# do a perceptron update of the parameters theta:
			for c1, c2 in sorted_v:
				w = [0] * len(c1.features)
				x = [c1j - c2j for c1j, c2j in zip(c1.features, c2.features)]
				# if theta * s1.features <= theta * s2.features:
				if sum([xj * wj for xj, wj in zip(x, w)]) <= 0:
					# mistakes += 1
					errors += 1
					# theta += eta * (s1.features - s2.features)
					w = [(opts.eta * xj) + wj for xj, wj in zip(x, w)]
				observations += 1
				if observations % 2000 == 0:
					sys.stderr.write(".")
		sys.stderr.write("classification error rate: %f (%d observation)\n" % (float(errors) / float(observations), observations))
	# return theta
	print "\n".join([str(weight) for weight in w])


















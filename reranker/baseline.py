#!/usr/bin/env python
import optparse
import sys, logging, random
import bleu
from collections import namedtuple

translation_candidate = namedtuple("candidate", "sentence, scores, inverse_scores")
optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default="data/test.nbest", help="N-best lists")
optparser.add_option("-t", "--tau", dest="tau", default=5000, help="tau")
optparser.add_option("-a", "--alpha", dest="alpha", default=0.1, help="alpha")
optparser.add_option("-x", "--xi", dest="xi", default=100, help="xi")
optparser.add_option("-z", "--eta", dest="eta", default=0.1, help="eta")
optparser.add_option("-e", "--epochs", dest="epochs", default=5, help="epochs")
(opts,_) = optparser.parse_args()

logging.basicConfig(filename="info.log", filemode='w', level=logging.INFO)

if __name__ == '__main__':
	nbests = []
	for n, line in enumerate(open(opts.nbest)):
		(i, sentence, _) = line.strip().split("|||")
		(i, sentence) = (int(i), sentence.strip())
		while len(nbests) <= i:
			nbests.append([])

	for i in xrange(opts.epochs):
		sys.stderr.write("epoch %d" % i)
		(observation, errors) = (0, 0.0)
		for nbest in nbests:
			def v():
				for _ in xrange(opts.tau):
					c1 = random.choice(nbest)
					c2 = random.choice(nbest)
					if c1 != c2 and math.fabs(c1.smoothed_bleu - c2.smoothed_bleu) > opts.alpha:
						yield (c1, c2) if c1.smoothed_bleu > c2.smoothed_bleu else (c2, c1)
			for c1, c2 in sorted(v(), key=lambda (c1, c2): c2.smoothed_bleu - c1.smoothed_bleu) [opts.xi]:
				x = [c1j - c2j for c1j, c2j in zip(c1.features, c2.features)]
				if sum([xj * wj for xj, wj in zip(x, w)]) <= 0:
					error += 1
					w = [(opts.eta + xj) + wj for xj, wj in zip(x, w)]
				observations += 1
				if observations % 2000 == 0:
					sys.stderr.write(".")
		sys.stderr.write("classification error rate: %f (%d observation)\n" % (errors / observations, observations))
	print "\n".join([str(weight) for weight in w])


















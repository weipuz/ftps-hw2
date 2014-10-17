#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default="debug.log", help="filename for logging output")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=10000, type="int", help="Number of sentences to use for training and alignment") #sys.maxint
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
	logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

k = 0
vf = opts.num_sents
ve = opts.num_sents
# Initialize t0 ## Easy choice: initialize uniformly ##
t = defaultdict(int)
for (f, e) in bitext:
	for (i, f_i) in enumerate(f):
		for (j, e_j) in enumerate(e):
			t[(f_i, e_j)] = 1.0 / float(len(f))
for i in range(0, 5):
	k += 1
	print "iteration", i
	# Initialize all counts to zero
	count_fe = defaultdict(int)
	count_e = defaultdict(int)
	# for each (f,e) in D
	for (f, e) in bitext:
		# for each fi in f
		for (i, f_i) in enumerate(f):
			# Z = 0 ## Z commonly denotes a normalization term ##
			z = 0
			# for each ej in e
			for (j, e_j) in enumerate(e):
				# Z += tk-1(fi|ej)
				z += t[(f_i, e_j)]
			# for each ej in e
			for (j, e_j) in enumerate(e):
				# c = tk-1(fi|ej)/Z
				c = float(t[(f_i, e_j)]) / float(z)
				# count(fi, ej) += c
				count_fe[(f_i, e_j)] += c
				# count(ej) += c
				count_e[e_j] += c
	# for each (f, e) in count
	for index, value in count_fe.iteritems():
		if 0.0 != count_e[index[1]]:
			# Set new parameters: tk(f|e) = count(f,e) / count(e)
			t[index] = float(count_fe[index]) / float(count_e[index[1]])

#logging.info(t)
# for each (f,e) in D
for (f, e) in bitext:
	# for each fi in f
	for (i, f_i) in enumerate(f):
		bestp = 0
		bestj = -1
		# for each ej in e
		for (j, e_j) in enumerate(e):
			# if t(fi|ej) > bestp
			if t[(f_i, e_j)] > bestp:
				# bestp = t(fi|ej)
				bestp = t[(f_i, e_j)]
				# bestj = j
				bestj = j
		if bestj >= 0:
			# align fi to ebestj
			print "%d-%d" % (i, bestj),
		else:
			print "%d-%d" % (i, 0),
	print

#Precision = 0.533823
#Recall = 0.704557
#AER = 0.407746







"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair. 

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""

import perc
import sys, optparse, os, codecs, time
from collections import defaultdict


def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    # insert your code here
    # please limit the number of iterations of training to n iterations
    return feat_vec

class ChunkTrainer():
	"perceptron trainer"

	def __init__(self):
		self.test_file = codecs.open("test.log", "w", "utf-8")
		#self.output_file = codecs.open("output.log", "w", "utf-8")
		self.feat_vec = defaultdict(float)

	# destructor
	def __exit__(self, type, value, traceback):
		#self.output_file.close()
		self.test_file.close()

	# turn a variable to a string for outputing
	def stringify(self, item):
		ans = ""
		if isinstance(item, (int, long, float, complex)):
			ans = repr(item)
		elif isinstance(item, (type)):
			ans = item.__name__
		elif isinstance(item, (defaultdict)):
			ans = type(item).__name__ + "("
			for index, value in item.iteritems():
				ans += self.stringify(index) + ": " + self.stringify(value) + ", "
			ans += ")"
		elif isinstance(item, (list, tuple, dict)):
			ans = type(item).__name__ + "("
			for index, value in enumerate(item):
				ans += self.stringify(index) + ": " + self.stringify(value) + ", "
			ans += ")"
		else:
			ans = item
		return ans

	# print the test comments
	def printTest(self, output = ""):
		self.test_file.write(self.stringify(output) + "\n")

	# call self to chunk
	def __call__(self, train_data, tagset, iter_num):
		self.tagset = tagset
		return self.chunk(train_data, tagset, iter_num)

	# learn chunking from train data
	def chunk(self, train_data, tagset, iter_num):
		# number of iteration
		iter_num = 10
		# size of training set
		train_size = 10000 #len(train_data)
		# error weight
		error_weight = 2.0
		# threshold for simplification
		simplify_threshold = 0.8
		print "iter_num", iter_num, "train_size", train_size, "error_weight", error_weight, "simplify_threshold", simplify_threshold
		self.printTest("iter_num" + repr(iter_num) + "train_size" + repr(train_size) + "error_weight" + repr(error_weight))

		# weight list for each item in the training set
		weight_list = [{}] * train_size
		error_list = [{}] * train_size
		weight_count = 0
		# assign initial weight value as one for train data
		for i in range(0, train_size):
			for j in range(0, len(train_data[i][0])):
				weight_count += 1
				weight_list[i][j] = 1.0

		# for each iteration
		for t in range(0, iter_num):
			print time.clock(), "iteration " + repr(t)
			self.printTest("========== iteration " + repr(t) + " ==========")

			count = 0
			# for each training set
			for i in range(0, train_size):
				#feat_list = train_data[i][1]
				labeled_list = train_data[i][0]
				# add features based on weights
				count += self.addFeatures(labeled_list, weight_list[i])
			print time.clock(), "add features done", count

			count = 0
			for i in range(0, train_size):
				labeled_list = train_data[i][0]
				# reduce features based on weights
				count += self.reduceFeatures(labeled_list, weight_list[i], error_list[i])
			print time.clock(), "reduce features done", count

			weight_count = 0
			for i in range(0, train_size):
				if 0 == i % 1000:
					print time.clock(), "learn from mistake", i, "/", train_size
				self.printTest("=== case " + repr(i) + " ===")
				labeled_list = train_data[i][0]
				feat_list = train_data[i][1]
				# update weights based on errors
				(w_list, w_count, e_list) = self.updateWeights(error_weight, labeled_list, feat_list)
				weight_list[i] = w_list
				weight_count += w_count
				error_list[i] = e_list
			print time.clock(), "learn features from mistake done", weight_count
			# simplify feat_vec
			self.feat_vec = self.simplifyFeatVec(simplify_threshold)
		return self.feat_vec

	# add features from labeled_list based on weight_list
	def addFeatures(self, labeled_list, weight_list):
		count = 0
		# for each item in labeled_list (training set)
		for index, line in enumerate(labeled_list):
			# ignore items with weight less than or equal to zero
			if (not (index in weight_list)) or weight_list[index] <= 0.0:
				continue
			count += 1
			weight = weight_list[index]
			# the word in position -2
			if index - 2 >= 0:
				neg2 = labeled_list[index - 2].split()
			else:
				neg2 = ["_B-2", "_B-2", "_B-2"]
			# the word in position -1
			if index - 1 >= 0:
				neg1 = labeled_list[index - 1].split()
			else:
				neg1 = ["_B-1", "_B-1", "_B-1"]
			# the word in position 0
			zero = labeled_list[index].split()
			# the word in position +1
			if index + 1 < len(labeled_list):
				pos1 = labeled_list[index + 1].split()
			else:
				pos1 = ["_B+1", "_B+1", "_B+1"]
			# the word in position +2
			if index + 2 < len(labeled_list):
				pos2 = labeled_list[index + 2].split()
			else:
				pos2 = ["_B+2", "_B+2", "_B+2"]
			# feature schema
			self.addFeat("U00", neg2[0], zero[2], weight)
			self.addFeat("U01", neg1[0], zero[2], weight)
			self.addFeat("U02", zero[0], zero[2], weight)
			self.addFeat("U03", pos1[0], zero[2], weight)
			self.addFeat("U04", pos2[0], zero[2], weight)
			self.addFeat("U05", neg1[0] + "/" + zero[0], zero[2], weight)
			self.addFeat("U06", zero[0] + "/" + pos1[0], zero[2], weight)
			self.addFeat("U10", neg2[1], zero[2], weight)
			self.addFeat("U11", neg1[1], zero[2], weight)
			self.addFeat("U12", zero[1] + "q", zero[2], weight)
			self.addFeat("U13", pos1[1], zero[2], weight)
			self.addFeat("U14", pos2[1], zero[2], weight)
			self.addFeat("U15", neg2[1] + "/" + neg1[1], zero[2], weight)
			self.addFeat("U16", neg1[1] + "/" + zero[1], zero[2], weight)
			self.addFeat("U17", zero[1] + "/" + pos1[1], zero[2], weight)
			self.addFeat("U18", pos1[1] + "/" + pos2[1], zero[2], weight)
			self.addFeat("U20", neg2[1] + "/" + neg1[1] + "/" + zero[1], zero[2], weight)
			self.addFeat("U21", neg1[1] + "/" + zero[1] + "/" + pos1[1], zero[2], weight)
			self.addFeat("U22", zero[1] + "/" + pos1[1] + "/" + pos2[1], zero[2], weight)
			self.addFeat("B", neg1[2] + "/" + zero[2], zero[2], weight)
		return count

	# reduce features from labeled_list based on weight_list
	def reduceFeatures(self, labeled_list, weight_list, error_list):
		#if None == error_list or 0 == len(error_list.keys()):
			#return 0
		count = 0
		# for each item in labeled_list
		for index, line in enumerate(labeled_list):
			# ignore items with weight less than or equal to zero
			if (not (index in weight_list)) or weight_list[index] <= 0.0:
				continue
			#error_list[index] = self.tagset # inject
			# ignore items with no error list
			if not (index in error_list) or None == error_list[index] or 0 == len(error_list[index]):
				continue
			count += 1
			weight = 1.0
			errors = error_list[index]
			if index - 2 >= 0:
				neg2 = labeled_list[index - 2].split()
			else:
				neg2 = ["_B-2", "_B-2", "_B-2"]
			if index - 1 >= 0:
				neg1 = labeled_list[index - 1].split()
			else:
				neg1 = ["_B-1", "_B-1", "_B-1"]
			zero = labeled_list[index].split()
			if index + 1 < len(labeled_list):
				pos1 = labeled_list[index + 1].split()
			else:
				pos1 = ["_B+1", "_B+1", "_B+1"]
			if index + 2 < len(labeled_list):
				pos2 = labeled_list[index + 2].split()
			else:
				pos2 = ["_B+2", "_B+2", "_B+2"]
			self.reduceFeat("U00", neg2[0], zero[2], weight, errors)
			self.reduceFeat("U01", neg1[0], zero[2], weight, errors)
			self.reduceFeat("U02", zero[0], zero[2], weight, errors)
			self.reduceFeat("U03", pos1[0], zero[2], weight, errors)
			self.reduceFeat("U04", pos2[0], zero[2], weight, errors)
			self.reduceFeat("U05", neg1[0] + "/" + zero[0], zero[2], weight, errors)
			self.reduceFeat("U06", zero[0] + "/" + pos1[0], zero[2], weight, errors)
			self.reduceFeat("U10", neg2[1], zero[2], weight, errors)
			self.reduceFeat("U11", neg1[1], zero[2], weight, errors)
			self.reduceFeat("U12", zero[1] + "q", zero[2], weight, errors)
			self.reduceFeat("U13", pos1[1], zero[2], weight, errors)
			self.reduceFeat("U14", pos2[1], zero[2], weight, errors)
			self.reduceFeat("U15", neg2[1] + "/" + neg1[1], zero[2], weight, errors)
			self.reduceFeat("U16", neg1[1] + "/" + zero[1], zero[2], weight, errors)
			self.reduceFeat("U17", zero[1] + "/" + pos1[1], zero[2], weight, errors)
			self.reduceFeat("U18", pos1[1] + "/" + pos2[1], zero[2], weight, errors)
			self.reduceFeat("U20", neg2[1] + "/" + neg1[1] + "/" + zero[1], zero[2], weight, errors)
			self.reduceFeat("U21", neg1[1] + "/" + zero[1] + "/" + pos1[1], zero[2], weight, errors)
			self.reduceFeat("U22", zero[1] + "/" + pos1[1] + "/" + pos2[1], zero[2], weight, errors)
			self.reduceFeat("B", neg1[2] + "/" + zero[2], zero[2], weight, errors)
		return count

	# update weights based on errors
	def updateWeights(self, error_weight, labeled_list, feat_list):
		# clear weight list
		weight_list = {}
		error_list = {}
		# reduce previous weights
		#for index in weight_list[i]:
			#weight_list[i][index] /= 3.0
			#if weight_list[i][index] < 0.1:
				#weight_list[i][index] = 0.0
			#else:
				#weight_count += 1
		# test by perc
		result = perc.perc_test(self.feat_vec, labeled_list, feat_list, self.tagset, self.tagset[0])
		weight_count = 0
		error_count = 0
		# for each result
		for index, value in enumerate(result):
			# correct labels from training set
			labeled = labeled_list[index].split()
			# if error
			if value != labeled[2]:
				error_count += 1
				# update weights
				if index in weight_list:
					weight_list[index] += error_weight
				else:
					weight_list[index] = error_weight
					weight_count += 1
				# update errors
				if index in error_list:
					error_list[index].append(value)
				else:
					error_list[index] = [value]
				self.printTest(index)
				#self.printTest(labeled_list[index])
				#self.printTest(value)
		# output test in conll format
		if error_count > 0:
			#test = "\n".join(perc.conll_format(result, labeled_list))
			#self.printTest(test)
			self.printTest("error_count " + repr(error_count))
		self.printTest()
		return (weight_list, weight_count, error_list)

	# simplify feat_vec
	def simplifyFeatVec(self, simplify_threshold):
		feat_vec1 = defaultdict(float)
		count = 0
		feat_sum = 0
		# for each item in feat_vec
		for index, value in self.feat_vec.iteritems():
			# keep items with value > 0
			if value > 0.0:
				count += 1
				feat_vec1[index] = value
				feat_sum += value
		feat_vec2 = defaultdict(float)
		feat_threshold = float(feat_sum) / float(count) * simplify_threshold
		count = 0
		for index, value in feat_vec1.iteritems():
			# keep items above the threshold
			if value >= feat_threshold:
				count += 1
				feat_vec2[index] = value
		print time.clock(), "simplify feature vector done", "threshold", feat_threshold, len(self.feat_vec), "=>", count
		return feat_vec2

	# add a feature
	def addFeat(self, name, schema, output, weight = 1.0):
		self.feat_vec[(name + ":" + schema, output)] += weight

	# reduce a feature
	def reduceFeat(self, name, schema, output, weight, error_list):
		weight = 1.0
		# for each tag
		for value in error_list:
			# if tag exists in feat_vec
			if value != output and ((name + ":" + schema, value) in self.feat_vec):
				# reduce features with different tags
				self.feat_vec[(name + ":" + schema, value)] -= weight

	# compare output and reference
	def compare(self):
		if not os.path.isfile("output.log"):
			return
		output = [ unicode(text.strip(), "utf-8") for text in open("output.log") ]
		reference = [ unicode(text.strip(), "utf-8") for text in open("data/reference250.txt") ]
		compare = codecs.open("compare.log", "w", "utf-8")
		count = 0
		output_index = 0
		for index in range(len(reference)):
			if "" == reference[index]:
				continue
			if output_index >= len(output):
				break
			while "" == output[output_index]:
				output_index += 1
				if output_index >= len(output):
					break
			if reference[index] != output[output_index]:
				count += 1
				compare.write("line: refer " + repr(index + 1) + " / output " + repr(output_index + 1) + "\noutput:\t" + output[output_index] + "\nrefer:\t" + reference[index] + "\n\n")
			output_index += 1
		compare.write("count: " + repr(count) + "\n")
		compare.close()
		print "comparing done"

	# deprecated
	def chunk0(self, labeled_list, feat_list, tagset, iter_num):
		self.printTest("iter_num: ")
		self.printTest(iter_num)
		self.printTest("tagset: ")
		self.printTest(tagset)
		self.printTest("labeled_list: ")
		self.printTest(labeled_list)
		self.printTest("feat_list: ")
		self.printTest(feat_list)
		self.printTest()

		## Data Structures ##

		# train: sentences with output labels: (w(j)[1:nj],t(j)[1:nj])
		# T: number of iterations over the training set. opts.n in default.py
		T = iter_num
		# phi: function that maps history/output-label pairs to d-dimensional feature vectors. phi for each history is provided in data/train.feats.gz
		# phi: Each component of the feature vector is called a feature function: phi s(h,t) where there are d feature functions, s=1,...,d.
		phi = feat_list
		d = len(phi)
		# w: Each sentence is a sequence of words and other useful information such as part-of-speech tags which we refer to as w[1:n].
		n = len(labeled_list)
		w = [0] * n
		# t: Each sequence of words is associated with a sequence of output labels t[1:n].
		t = [None] * n
		# h: At each point there is a history which is the context in which the output label is assigned to a particular word wi. A history is a three-tuple
		h = ()
		# Phi: global feature vector defined as above by summing over all local feature vectors phi
		Phi = [0] * len(labeled_list)
		for i in range(0, n - 1):
			Phi[i] = 0
			for j in range(0, n - 1):
				Phi[i] += phi[j]
		# w: d dimensional weight vector. one weight for each feature in the feature vector.

		## Initialization ##

		# Set weight vector w to zeroes.
		default_tag = tagset[0]

		## Main Loop ##

		# for t = 1, ..., T, for j = 1, ..., n
		#for t in range(0, T):
			#for j in range(0, len(labeled_list)):
				# Use the Viterbi algorithm to find the output of the model on the i-th training sentence (the function perc_test in perc.py implements the Viterbi algorithm) where Tnj is the set of all tag sequences of length nj.
		w = perc.perc_test(w, labeled_list, feat_list, tagset, default_tag)
		self.printTest("w: ")
		self.printTest("\n".join(conll_format(w, labeled_list)))
		return w

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    ct = ChunkTrainer()
    ct.compare()
    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."
    #
    feat_vec = ct(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)





























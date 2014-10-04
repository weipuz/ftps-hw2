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
import sys, optparse, os, codecs
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
		self.feat_vec = defaultdict(int)

	# destructor
	def __exit__(self, type, value, traceback):
		self.test_file.close()
		#self.output_file.close()

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

	def __call__(self, train_data, tagset, iter_num):
		for i in range(0, len(train_data)):
			labeled_list = train_data[i][0]
			feat_list = train_data[i][1]
			#for feat in feat_list:
				#if not(feat in self.feat_vec):
					#self.feat_vec[feat] = 0
			self.addFeatures(labeled_list)
			self.changeFeatures(labeled_list)
		self.feat_vec = self.simplifyFeatVec()
		self.printTest(self.feat_vec)
		return self.feat_vec

	def addFeatures(self, labeled_list):
		for index, line in enumerate(labeled_list):
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
			self.addFeat("U00", neg2[0], zero[2], tagset)
			self.addFeat("U01", neg1[0], zero[2], tagset)
			self.addFeat("U02", zero[0], zero[2], tagset)
			self.addFeat("U03", pos1[0], zero[2], tagset)
			self.addFeat("U04", pos2[0], zero[2], tagset)
			self.addFeat("U05", neg1[0] + "/" + zero[0], zero[2], tagset)
			self.addFeat("U06", zero[0] + "/" + pos1[0], zero[2], tagset)
			self.addFeat("U10", neg2[1], zero[2], tagset)
			self.addFeat("U11", neg1[1], zero[2], tagset)
			self.addFeat("U12", zero[1] + "q", zero[2], tagset)
			self.addFeat("U13", pos1[1], zero[2], tagset)
			self.addFeat("U14", pos2[1], zero[2], tagset)
			self.addFeat("U15", neg2[1] + "/" + neg1[1], zero[2], tagset)
			self.addFeat("U16", neg1[1] + "/" + zero[1], zero[2], tagset)
			self.addFeat("U17", zero[1] + "/" + pos1[1], zero[2], tagset)
			self.addFeat("U18", pos1[1] + "/" + pos2[1], zero[2], tagset)
			self.addFeat("U20", neg2[1] + "/" + neg1[1] + "/" + zero[1], zero[2], tagset)
			self.addFeat("U21", neg1[1] + "/" + zero[1] + "/" + pos1[1], zero[2], tagset)
			self.addFeat("U22", zero[1] + "/" + pos1[1] + "/" + pos2[1], zero[2], tagset)
			self.addFeat("B", neg1[2] + "/" + zero[2], zero[2], tagset)

	def changeFeatures(self, labeled_list):
		for index, line in enumerate(labeled_list):
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
			self.changeFeat("U00", neg2[0], zero[2], tagset)
			self.changeFeat("U01", neg1[0], zero[2], tagset)
			self.changeFeat("U02", zero[0], zero[2], tagset)
			self.changeFeat("U03", pos1[0], zero[2], tagset)
			self.changeFeat("U04", pos2[0], zero[2], tagset)
			self.changeFeat("U05", neg1[0] + "/" + zero[0], zero[2], tagset)
			self.changeFeat("U06", zero[0] + "/" + pos1[0], zero[2], tagset)
			self.changeFeat("U10", neg2[1], zero[2], tagset)
			self.changeFeat("U11", neg1[1], zero[2], tagset)
			self.changeFeat("U12", zero[1] + "q", zero[2], tagset)
			self.changeFeat("U13", pos1[1], zero[2], tagset)
			self.changeFeat("U14", pos2[1], zero[2], tagset)
			self.changeFeat("U15", neg2[1] + "/" + neg1[1], zero[2], tagset)
			self.changeFeat("U16", neg1[1] + "/" + zero[1], zero[2], tagset)
			self.changeFeat("U17", zero[1] + "/" + pos1[1], zero[2], tagset)
			self.changeFeat("U18", pos1[1] + "/" + pos2[1], zero[2], tagset)
			self.changeFeat("U20", neg2[1] + "/" + neg1[1] + "/" + zero[1], zero[2], tagset)
			self.changeFeat("U21", neg1[1] + "/" + zero[1] + "/" + pos1[1], zero[2], tagset)
			self.changeFeat("U22", zero[1] + "/" + pos1[1] + "/" + pos2[1], zero[2], tagset)
			self.changeFeat("B", neg1[2] + "/" + zero[2], zero[2], tagset)

	def simplifyFeatVec(self):
		feat_vec1 = defaultdict(int)
		for index, value in self.feat_vec.iteritems():
			if value > 0:
				feat_vec1[index] = value
		return feat_vec1

	def addFeat(self, name, schema, output, tagset):
		self.feat_vec[(name + ":" + schema, output)] += 1

	def changeFeat(self, name, schema, output, tagset):
		for value in tagset:
			if value != output and ((name + ":" + schema, value) in self.feat_vec):
				self.feat_vec[(name + ":" + schema, value)] -= 1

	def compare(self):
		output = [ unicode(text.strip(), "utf-8") for text in open("output.log") ]
		reference = [ unicode(text.strip(), "utf-8") for text in open("data/reference250.txt") ]
		compare = codecs.open("compare.log", "w", "utf-8")
		count = 0
		output_index = 0
		for index in range(len(reference)):
			if "" == reference[index]:
				continue
			while "" == output[output_index]:
				output_index += 1
			if reference[index] != output[output_index]:
				count += 1
				compare.write("line: refer " + repr(index + 1) + " / output " + repr(output_index + 1) + "\noutput:\t" + output[output_index] + "\nrefer:\t" + reference[index] + "\n\n")
			output_index += 1
		compare.write("count: " + repr(count) + "\n")
		compare.close()
		print "comparing done"

	def chunk(self, labeled_list, feat_list, tagset, iter_num):
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

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

def perc_train(train_data, tagset, n):
    feat_vec = defaultdict(int)
    # insert your code here
    ct = ChunkTrainer()
    feat_vec = ct(train_data, tagset, n)
    # please limit the number of iterations of training to n iterations
    return feat_vec

class ChunkTrainer():
	"perceptron trainer"

	def __init__(self):
		self.test_file = codecs.open("test.log", "w", "utf-8")
		self.output_file = codecs.open("output.log", "w", "utf-8")

	# destructor
	def __exit__(self, type, value, traceback):
		self.output_file.close()
		self.test_file.close()

	def stringify(self, item):
		ans = ""
		if isinstance(item, (int, long, float, complex)):
			ans = repr(item)
		elif isinstance(item, (list, tuple, dict)):
			ans = "("
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
		return self.chunk(train_data[0][0], train_data[0][1], tagset, iter_num)

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
		# phi: Each component of the feature vector is called a feature function: phi s(h,t) where there are d feature functions, s=1,…,d.
		phi = feat_list
		d = len(phi)
		# w: Each sentence is a sequence of words and other useful information such as part-of-speech tags which we refer to as w[1:n].
		n = len(labeled_list)
		w = [0] * n
		# t: Each sequence of words is associated with a sequence of output labels t[1:n].
		t = [None] * n
		# h: At each point there is a history which is the context in which the output label is assigned to a particular word wi. A history is a three-tuple: h=(t−1,w[1:n],i), where t−1 is the output label for wi−1.
		h = ()
		# Phi: global feature vector defined as above by summing over all local feature vectors phi
		Phi = [0] * len(labeled_list)
		for i in range(0, n - 1):
			Phi[i] = 0
			for j in range(0, n - 1):
				Phi[i] += phi[]
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

if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-n", "--numiterations", dest="n", default=int(10), help="number of iterations of training")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."
    feat_vec = perc_train(train_data, tagset, int(opts.n))
    perc.perc_write_to_file(feat_vec, opts.modelfile)


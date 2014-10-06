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
import sys, optparse, os, codecs, time, types
from collections import defaultdict

def perc_train_initial(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    train_data_size = len(train_data)
    # initial the feat_vec to summarize all the features of the words in all the sentences
    for i in range(0,train_data_size):
         for k in range(0,len(train_data[i][0])):
              for j in range(0,20):           
                current_feat = train_data[i][1]
                current_labeled_list = train_data[i][0]
                current_labeled = current_labeled_list[k]
                current_labeled = ''.join(current_labeled)
                current_tag = current_labeled.split()[2]
                #print current_tag
                feat_vec[current_feat[j+20*k],current_tag] +=1
    # 10 iteration
    for s in range(0,numepochs):
        feat_vec = perc_train(feat_vec,train_data, tagset, numepochs)
    return feat_vec

def perc_train_without_initial(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    # 10 iteration
    for s in range(0,numepochs):
        feat_vec = perc_train(feat_vec,train_data, tagset, numepochs)
    return feat_vec
    return feat_vec

def perc_train(feat_vec,train_data, tagset, numepochs):
    default_tag = tagset[0]
    #get new output
    for m in range(0,len(train_data)):
    	result = perc.perc_test(feat_vec, train_data[m][0], train_data[m][1], tagset, default_tag)    
        ture_result = []
        current_labeled_list = train_data[m][0]
        #combine all target in one sentence
        for l in range(0,len(train_data[m][0])):
        	current_labeled = current_labeled_list[l]
         	current_labeled = ''.join(current_labeled)
          	current_tag = current_labeled.split()[2]
          	ture_result.append(current_tag)
        
        error_count =0
        error_index =[]
        for n in range(0,len(ture_result)):
        	if result[n] != ture_result[n]:
        		error_count +=1
        		error_index.append(n)
        #print error_index
        for p in range(0,len(error_index)):
        	currrent_index = error_index[p]
        	feat_vec = update(currrent_index,result[currrent_index],ture_result[currrent_index],feat_vec,train_data[m][0],train_data[m][1])
        	
    return feat_vec
   
def update(currrent_index,result,ture_result,feat_vec,label_list,feat_list):
	for j in range(0,20):       	
		feat_vec[feat_list[j+20*currrent_index],result] -= 1
		feat_vec[feat_list[j+20*currrent_index],ture_result] += 1
	return feat_vec

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(7), help="number of epochs of training; in each epoch we iterate over over all the training examples")
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
    #feat_vec =perc_train_initial(train_data, tagset, int(opts.numepochs))
    feat_vec =perc_train_without_initial(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)
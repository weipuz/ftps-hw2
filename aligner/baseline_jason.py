#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
ef_count = defaultdict(int)
#print bitext[5]



for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
		fe_count[(f_i,e_j)] += 1
     
  for e_j in set(e):
	e_count[e_j] += 1
	for f_i in set(f):
		ef_count[(e_j,f_i)] += 1
  if n % 500 == 0:  
    sys.stderr.write(".")

#print f_count.items()
#print fe_count.items()

# Initialize t0 ## Easy choice: initialize uniformly ##
V_f = float(1.0/len(f_count))
V_e = float(1.0/len(e_count))
t_fe = defaultdict(int)
t_ef = defaultdict(int)
for key in set(fe_count.keys()):
    t_fe[key] = V_f
for key in set(ef_count.keys()):
    t_ef[key] = V_e
#print fe_count

#add null

for key in set(e_count.keys()):
    t_fe[(None,key)] = V_f
   
for key in set(f_count.keys()):
    t_ef[(None,key)] = V_e

'''
for key in set(t_fe.keys()):
    if key[0] == None:
        print key

print ('stop')
'''
#test = fe_count.keys()
#print test
#print len(fe_count)
#print len(t_fe)
iter_num = 8
lamda=0.005 # the n in add-n smoothing
lamda2=0.005 # add n2 for second model add-n smoothing
B = 0 # the addup factor for the V in add-n smoothing
#d = 5 # #not using##the max distance between two possible alignment #
#print ('down.....')
#print len(fe_count)
#print ('....')


########train fe model	
for k in range(iter_num):
    k +=1
    #Initialize all counts to zero
    count_e = defaultdict(int)
    count_fe = defaultdict(int)
    test_count = 0
    for (n, (f, e)) in enumerate(bitext):
        f_index =0
        for f_word in set(f):
            #Z = 0 ## Z commonly denotes a normalization term ##
            Z = 0
            #test_count_1 =0
			
            for e_word in set(e):
				if f_index ==0:
					Z += t_fe[(None,e_word)]                                          
				Z += t_fe[(f_word,e_word)]
            for e_word_ in set(e):
				if f_index ==0:
					c = float(t_fe[(None,e_word_)]/Z)
					count_fe[(None,e_word_)] += c
					count_e[e_word_] += c
			  
				c = float(t_fe[(f_word,e_word_)]/Z)
				#print c
				count_fe[(f_word,e_word_)] += c
				count_e[e_word_] += c 
            f_index += 1
        if n % 500 == 0:  
            sys.stderr.write(".")
    #print test_count
    #test_count +=1
    #print count_e.viewvalues()
    #print ('down down...')
	
	V = len(e_count)+B
	
    for fe_pair in set(count_fe.keys()):
        #if fe_pair[1] == None:
        #    print fe_pair
        #print ('stop')
        
        t_fe[fe_pair] = float((count_fe[fe_pair]+lamda)/(count_e[fe_pair[1]]+lamda*V))
        test_count +=1
        if test_count % 100000 == 0:
            sys.stderr.write(".")
        #sys.stderr.write(".")
    #print len(t_fe)
    #print ('..............')
    #print len(count_fe)


## train  ef model	
k=0
for k in range(iter_num):
    k +=1
    #Initialize all counts to zero
    count_f = defaultdict(int)
    count_ef = defaultdict(int)
    test_count = 0
    for (n, (f, e)) in enumerate(bitext):
        e_index =0
        for e_word in set(e):
            #Z = 0 ## Z commonly denotes a normalization term ##
            Z = 0
            #test_count_1 =0
          
            
            for f_word in set(f):
                if e_index ==0:
                    Z += t_ef[(None,f_word)]
                Z += t_ef[(e_word,f_word)]
            
            for f_word_ in set(f):
				if e_index ==0:
					c = t_ef[(None,f_word_)]/Z
					count_ef[(None,f_word_)] += c
					count_f[f_word_] += c
				c = t_ef[(e_word,f_word_)]/Z
				#print c
				count_ef[(e_word,f_word_)] += c
				count_f[f_word_] += c 
            e_index += 1

        if n % 500 == 0:  
            sys.stderr.write(".")
    #print test_count
    #test_count +=1
    #print count_e.viewvalues()
    #print ('down down...')
	
	V = len(f_count)+B
    for ef_pair in set(count_ef.keys()):
        #if fe_pair[1] == None:
        #    print fe_pair
        #print ('stop')
        
        t_ef[ef_pair] = float((count_ef[ef_pair]+lamda2)/(count_f[ef_pair[1]]+lamda2*V))
        test_count +=1
        if test_count % 100000 == 0:
            sys.stderr.write(".")	
	

	
#print t_fe 

def checkalign(i,j,alignment):
		align = "%i-%i " % (i,j)
		if align in alignment:
			return True
		else: 
			return False


### decode use two model    
for (f,e) in bitext:
	fe_list=[]
	ef_list=[]
	
	for (i, f_i) in enumerate(f):
		bestp = 0
		bestj = 0
		for (j, e_j) in enumerate(e):            
			if t_fe[(f_i,e_j)] > bestp: #and abs(i-j ) < d:
				bestp = t_fe[(f_i,e_j)]
				bestj =j
		#if t_fe[(None,f_i)] < bestp:
		fe_list.append("%i-%i " % (i,bestj))
			
		#if abs(i-bestj ) < 10:
		#		fe_list.append("%i-%i " % (i,bestj))
			
	for (i, e_i) in enumerate(e):
		bestp = 0
		bestj = 0
		for (j, f_j) in enumerate(f):            
			if t_ef[(e_i,f_j)] > bestp:# and abs(i-j ) < d:
				bestp = t_ef[(e_i,f_j)]
				bestj =j
		#if t_ef[(None,e_i)] < bestp:
		ef_list.append("%i-%i " % (bestj,i)) 
		#if abs(i-bestj ) < 10:
			#ef_list.append("%i-%i " % (bestj,i)) 		
		
	intersect = [val for val in fe_list if val in ef_list] 
	union = fe_list + ef_list
	
	
		
	#####Add growing step
	neighboring = ((-1,0),(0,-1),(1,0),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
	newadd = True
	#grow-diag:
	align = list([tuple(map(int, x.split("-"))) for x in intersect])
	#fe_align = list([tuple(map(int, x.split("-"))) for x in fe_list])
	#ef_align = list([tuple(map(int, x.split("-"))) for x in ef_list])
	union = set([tuple(map(int, x.split("-"))) for x in union])
	while newadd:	
		newadd=False
		for (i, e_i) in enumerate(e):
			for (j, f_j) in enumerate(f):	
				if (i, j) in align:
					#print i, j
					for (i_d, j_d) in neighboring:
						#print i,j,i_d,j_d
						i_new = min(max(0,i+i_d),len(e)-1)
						j_new = min(max(0,j+j_d),len(f)-1)
						if (i_new, j_new) in union:
							if (i_new not in [a[0] for a in align]) or (j_new not in [a[1] for a in align]):
								newadd = True
								align.append((i_new,j_new))
								#print i_new,j_new
		if not newadd: break
	'''	
	##final(e2f):
	for i_new in range(0,len(e)):
		for j_new in range(0,len(f)):	
			if (i_new, j_new) in ef_align:
				if (i_new not in [a[0] for a in align]) or (j_new not in [a[1] for a in align]):
					align.append((i_new,j_new))
	##final(f2e):
	for i_new in range(0,len(e)):
		for j_new in range(0,len(f)):	
			if (i_new, j_new) in fe_align:
				if (i_new not in [a[0] for a in align]) or (j_new not in [a[1] for a in align]):
					align.append((i_new,j_new))
	'''
	alignment = sorted(align)
	for (i, j) in alignment:
		sys.stdout.write("%i-%i " % (i,j))
	
	#for item in intersect:
	#	sys.stdout.write(item)      
		
	sys.stdout.write("\n")

		
	
'''

### decode use f_e model    
for (f,e) in bitext:
    for (i, f_i) in enumerate(f):
        bestp = 0
        bestj = 0
		
        for (j, e_j) in enumerate(e):            
            if t_fe[(f_i,e_j)] > bestp and abs(i-j ) < d:
                bestp = t_fe[(f_i,e_j)]
                bestj =j
        #if t_fe[(None),e_j] < bestp:
        sys.stdout.write("%i-%i " % (i,bestj))
            
        #if abs(i-bestj ) < 10:
            #print i-bestj
            #sys.stdout.write("%i-%i " % (i,bestj))
               
    sys.stdout.write("\n")
	

### decode use e_f model 
for (f,e) in bitext:
    for (i, e_i) in enumerate(e):
        bestp = 0
        bestj = 0
        for (j, f_j) in enumerate(f):            
            if t_ef[(e_i,f_j)] > bestp and abs(i-j ) < d:
                bestp = t_ef[(e_i,f_j)]
                bestj =j
        if t_ef[(None,f_j)] < bestp:
            sys.stdout.write("%i-%i " % (bestj,i))
            
        #if abs(i-bestj ) < 10:
            #print i-bestj
            #sys.stdout.write("%i-%i " % (i,bestj))
            
            
        
        
        
        
    sys.stdout.write("\n")
'''    
'''
dice = defaultdict(int)
for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
  dice[(f_i,e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])
  if k % 5000 == 0:
    sys.stderr.write(".")
sys.stderr.write("\n")
#print dice.items()


for (f, e) in bitext:
  for (i, f_i) in enumerate(f): 
    for (j, e_j) in enumerate(e):
      if dice[(f_i,e_j)] >= opts.threshold:
        sys.stdout.write("%i-%i " % (i,j))
  sys.stdout.write("\n")
'''
import os
from numpy import genfromtxt
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csc_matrix, lil_matrix
import multiprocessing as mp

def load_stopwds(filename):
	with open(filename, "r") as fin:
		tmp = fin.read().splitlines()
	
	return tmp

def load_labels(label):
	'''
	load the label csv file

	:param label: the path to the label file
	:return: a list with all the filenames, and a numpy array of all labels
	'''
	tmp = genfromtxt(label, delimiter = ",", dtype = str)

	return [_[0] for _ in tmp[1:]], np.array([_[3] for _ in tmp[1:]])

def load_corpus(corpus_dir, filelist):
	'''
	load all essays into a list

	:param corpus_dir: the directory of all the essays.
	:param filelist: a list of all file names w/o extension.
	:return: a list of all the essays. each element of the list corresponds to one corpus.
	'''

	tmp = []

	for f in filelist:
		with open(os.path.join(corpus_dir, f + ".txt"), "r") as fin:
			tmp.append(fin.read())

	return tmp

def word_ng(corpus, mean_norm, **kwargs):
	'''
	compute word based ngram model

	:param corpus: a list of all the corpuses. each element of the list corresponds to one corpus.
	:param mean_norm: a boolean enables mean normalization.
	:param kwargs: keyword arguments to be passed into CountVectorizer().
	:return: feature matrix and the vectorizer.
	'''
	vectorizer = CountVectorizer(**kwargs)
	#vectorizer = TfidfVectorizer(**kwargs)

	## returns Compressed Sparse Row matrix
	x = vectorizer.fit_transform(corpus)

	if mean_norm:
		x = preprocessing.scale(x, with_mean=False)

	tmp = x

	#min_max_scaler = preprocessing.MinMaxScaler()
	#tmp = min_max_scaler.fit_transform(tmp)

	#tmp = preprocessing.normalize(tmp)

	## a parallel implementation of recurrence word ngram
	# if rec:
	# 	#print("start of //")
	# 	nproc = 10

	# 	csc = csc_matrix(tmp)
	# 	l = csc.shape[1]

	# 	curr = 0
	# 	chunk_size = l // nproc
	# 	q = mp.Manager().Queue()
	# 	procs = []
	# 	for i in range(nproc):
	# 		args = (csc, curr, (lambda: curr + chunk_size if curr + chunk_size <= l else l)(), q)
	# 		curr += chunk_size
	# 		p = mp.Process(target = recurrence, args = args)
	# 		p.start()
	# 		procs.append(p)

	# 	for proc in procs:
	# 		proc.join()
	# 	#print("start of +")
	# 	z = q.get()
	# 	while not q.empty():
	# 		z = np.add(z, q.get())

	# 	#print("end of +\nend of //\nstart of dot")	
	# 	tmp = np.dot(csc, z.T)
	# 	#print("end of dot")


	return tmp, vectorizer

## to be called in each process
# def recurrence(sparse_matrix, i, j, q):
# 	'''
# 	couting the recurrence of word ngram, and generating a dia-sparse matrix

# 	:param sparse_matrix: original sparse ngram feature matrix
# 	:param i: start index
# 	:param j: end index
# 	:param q: Queue
# 	:return:
# 	'''
# 	pid = mp.current_process().name
# 	#print(pid, "start")

# 	l = sparse_matrix.shape[1]
# 	z = lil_matrix((l, l))
# 	try:
# 		for x in range(i, j):
# 			if sparse_matrix[:, x].count_nonzero() > 1:
# 				z[x, x] = 1
# 	except:
# 		pass
# 	finally:
# 		q.put(z)

# 	#print(pid, "finish")

# def precision(pred, label):
# 	'''
# 	return the precision of prediction
# 	:param pred: an array of prediction
# 	:param label: an array of labels
# 	:return: precision
# 	'''
# 	assert(len(pred) == len(label))

# 	l = len(pred)

# 	tp = 0
# 	for i in range(l):
# 		if pred[i] == label[i]:
# 			tp += 1

# 	return tp / l

def ngram_range(i, j, l):
	x = range(i, j + 1)
	tmp = []
	for _ in range(l + 1):
		for __ in range(len(x)):
				try:
					a = x[__]
					b = x[__ + _ - 1]
					if a <= b:
						tmp.append((a,b))
				except:
					pass

	return tmp[1:]	
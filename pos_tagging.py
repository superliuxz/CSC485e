import os, sys
import nltk
from nltk.corpus import brown
from nltk.tag import UnigramTagger
from simplify_tagset import simplify_tagset
from multiprocessing.dummy import Pool
import itertools

def tag_essay(path_to_file, simplify = "F"):

	with open(path_to_file, "r") as fin:
		tokenized_essay = fin.read().split()
		# tokenized_essay = nltk.word_tokenize(fin.read())

	tokenized_essay = [word.lower().replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace(";", "").replace(":", "").replace('"', "") for word in tokenized_essay]

	tagger = UnigramTagger(brown.tagged_sents())

	with open(path_to_file + ".pos", "w") as fout:

		tagged_words = tagger.tag(tokenized_essay)

		if simplify == "T": tagged_words = simplify_tagset(tagged_words)

		nword = 0
		typos = 0
		for word, tag in tagged_words:
			if word != "" : 
				nword += 1
				if tag == "None": typos += 1
				fout.write(word + ' , ' + str(tag) + "\n")

	with open(path_to_file + ".typos", "w") as fout:
		fout.write("{} , {}".format(typos, typos / nword))

	print("finish {}".format(path_to_file))				

# tag_essay("/Users/will/Desktop/csc485e/nli-shared-task-2017/data/essays/train/original/00001.txt")

if __name__ == "__main__":

	GNU_PARALLEL = True

	if GNU_PARALLEL: 
		tag_essay(sys.argv[1], sys.argv[2])

	## the following parallelism is very slow (compared with GNU parallel) for unknown reason
	## stackoverflow does suggest avoiding multithreading for CPU intensive jobs but i am using multiprocessing ...	
	else:	
		n_thread = int(sys.argv[1])

		path_to_essays = "data/essays/train/original/"
		essays = os.listdir(path_to_essays)

		essays = list(map(lambda x: path_to_essays + x, filter(lambda x: x.endswith(".txt"), essays)))

		pool = Pool(n_thread) 

		pool.map(tag_essay, essays)

		pool.close()
		pool.join()



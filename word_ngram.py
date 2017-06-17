import numpy as np
import nli_util as nu
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def word_ngram():
	stopwords = nu.load_stopwds("stopwords.txt")

	training_filelist, training_labels = nu.load_labels("data/labels/train/labels.train.csv")
	pred_filelist, pred_labels = nu.load_labels("data/labels/dev/labels.dev.csv")

	training_corpus = nu.load_corpus("data/essays/train/original", training_filelist)
	pred_corpus = nu.load_corpus("data/essays/dev/original", pred_filelist)

	for i in range(1, 4):
		for n in range(4 - i):
			print(n+1, n+i)
			training_features, vectorizer = nu.word_ng(training_corpus, False, ngram_range = (n+1, n+i), stop_words = None, binary = True, min_df = 1)

			pred_features = vectorizer.transform(pred_corpus)

			#clf = LinearSVC()
			clf = LogisticRegression(n_jobs = -1)
			#print("start of fit")
			clf.fit(training_features, training_labels)
			#print("end of fit\nstart of pred")
			prediction = clf.predict(pred_features)
			#print("end of pred")
			print(nu.precision(prediction, pred_labels))

if __name__ == "__main__":
	word_ngram()

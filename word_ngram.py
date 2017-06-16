import numpy as np
import nli_util as nu
from sklearn.svm import LinearSVC

def word_ngram():
	stopwords = nu.load_stopwds("stopwords.txt")

	training_filelist, training_labels = nu.load_labels("data/labels/train/labels.train.csv")
	pred_filelist, pred_labels = nu.load_labels("data/labels/dev/labels.dev.csv")

	training_corpus = nu.load_corpus("data/essays/train/original", training_filelist)
	pred_corpus = nu.load_corpus("data/essays/dev/original", pred_filelist)

	for i in range(1, 4):
		for n in range(6 - i):
			training_features, vectorizer = nu.word_ng(training_corpus, False, True, ngram_range = (n+1, n+i), dtype = np.float64, stop_words = None, binary = False)

			pred_features = vectorizer.transform(pred_corpus)

			clf = LinearSVC()
			clf.fit(training_features, training_labels)
			prediction = clf.predict(pred_features)

			print(nu.precision(prediction, pred_labels))

if __name__ == "__main__":
	word_ngram()

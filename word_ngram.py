import numpy as np
import nli_util as nu
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

WRITE_SUBMISSION = True
def word_ngram():
	# stopwords = nu.load_stopwds("stopwords.txt")

	training_filelist, training_labels = nu.load_labels("data/labels/train/labels.train.csv")
	pred_filelist, pred_labels = nu.load_labels("data/labels/dev/labels.dev.csv")

	training_corpus = nu.load_corpus("data/essays/train/original", training_filelist)
	pred_corpus = nu.load_corpus("data/essays/dev/original", pred_filelist)

	# ngram_ranges = nu.ngram_range(1,3,3)
	# for ng_range in ngram_ranges:
	#	print("===========", ng_range, "===========")
	training_features, vectorizer = nu.word_ng(training_corpus, False, ngram_range = (1, 3), stop_words = None, analyzer = "word", binary = True, min_df = 2)

	pred_features = vectorizer.transform(pred_corpus)

	clf = LinearSVC()
	#clf = LogisticRegression(n_jobs = -1)
	#print("start of fit")
	clf.fit(training_features, training_labels)
	#print("end of fit\nstart of pred")
	prediction = clf.predict(pred_features)
	#print("end of pred")

	print("Summary:\n", metrics.classification_report(pred_labels, prediction))
	print("Consusion matrix:\n", metrics.confusion_matrix(pred_labels, prediction), "\n")
	print("precision:", np.mean(prediction == pred_labels), "\n")

	if WRITE_SUBMISSION:
		assert( len(prediction) == len(pred_filelist))
		with open("output.csv", "w") as fout:
			fout.write("prediction,test_taker_id\n")
			for i in range(len(pred_filelist)):
				fout.write(prediction[i] + "," + pred_filelist[i] + "\n")


if __name__ == "__main__":
	word_ngram()

import numpy as np
import nli_util as nu
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics

WRITE_SUBMISSION = False
def word_ngram():
	# stopwords = nu.load_stopwds("stopwords.txt")

	training_filelist, training_labels = nu.load_labels("data/labels/train/labels.train.csv")
	pred_filelist, pred_labels = nu.load_labels("data/labels/dev/labels.dev.csv")

	training_corpus2 = nu.load_corpus("data/essays/train/pos", training_filelist)
	pred_corpus2 = nu.load_corpus("data/essays/dev/pos", pred_filelist)

	training_corpus = nu.load_corpus("data/essays/train/original", training_filelist)
	pred_corpus = nu.load_corpus("data/essays/dev/original", pred_filelist)

	# ngram_ranges = nu.ngram_range(1,5,3)
	# for ng_range in ngram_ranges:

	# 	print("===========", ng_range, "===========")

	for i in range(1, 11):		
	
		training_features, vectorizer = nu.word_ng(training_corpus, False, ngram_range = (1, 3), \
			stop_words = None, analyzer = "word", min_df = 2, binary = True)

		pred_features = vectorizer.transform(pred_corpus)

		training_features2, vectorizer2 = nu.word_ng(training_corpus2, False, ngram_range = (2, 4), \
			stop_words = None, analyzer = "word", min_df = 1, binary = False)

		pred_features2 = vectorizer2.transform(pred_corpus2)

		training_features = nu.concat(training_features, training_features2, i/10)
		pred_features = nu.concat(pred_features, pred_features2, i/10)

		clf = LinearSVC()
		#clf = SVC(kernel="poly")
		#clf = LogisticRegression(n_jobs = -1, solver = "sag")
		#clf = SGDClassifier(loss = "hinge", n_jobs = -1, shuffle = True)

		clf.fit(training_features, training_labels)
		prediction = clf.predict(pred_features)

		# print("Summary:\n", metrics.classification_report(pred_labels, prediction))
		# print("Consusion matrix:\n", metrics.confusion_matrix(pred_labels, prediction,\
		#	 labels=["GER", "ARA", "TUR", "HIN", "TEL", "SPA", "FRE", "ITA", "CHI", "JPN", "KOR"]), "\n")

		print("accuracy:", np.mean(prediction == pred_labels), "\n")

		# if WRITE_SUBMISSION:
		# 	assert( len(prediction) == len(pred_filelist))
		# 	with open("output.csv", "w") as fout:
		# 		fout.write("prediction,test_taker_id\n")
		# 		for i in range(len(pred_filelist)):
		# 			fout.write(prediction[i] + "," + pred_filelist[i] + "\n")


if __name__ == "__main__":
	word_ngram()

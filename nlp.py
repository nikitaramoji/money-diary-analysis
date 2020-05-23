"""
Natural language processing assignment from Brown University's CSCI 1951A
(Data Science) class

Created by Alex Jang (ajang)
"""

import os
import sys
import nltk
import sklearn

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

# Downloads the NLTK stopword corpus if not already downloaded
try:
	nltk.data.find('corpora/stopwords')
except LookupError:
	nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

# sklearn modules for data processing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# sklearn modules for LSA
from sklearn.decomposition import TruncatedSVD

# sklearn modules for classification
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# sklearn modules for clustering
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D # Not used but needed to make 3D plots

def process_document(text):
	"""
	Processes a text document by converting all words to lower case,
	tokenizing, removing all non-alphabetical characters,
	and stemming each word.

	Args:
		text: A string of the text of a single document.

	Returns:
		A list of processed words from the document.
	"""
	# Convert words to lower case
	text = text.lower()

	# TODO: Tokenize document and remove all non-alphanumeric characters
	tokenizer = RegexpTokenizer(r'\w+')
	text = tokenizer.tokenize(text)

	processed = []
	# TODO: Remove stopwords
	stop_words = stopwords.words('english')
	for w in text:
		if w not in stop_words:
			processed.append(w)
	# TODO: Stem words
	stemmer = SnowballStemmer('english')
	for i in range(0, len(processed)):
		processed[i] = stemmer.stem(processed[i])

	# TODO: Return list of processed words
	return processed


def read_data(data_dir):
	"""
	Preprocesses all of the text documents in a directory.

	Args:
		data_dir: the directory to the data to be processed
	Returns:
		1. A mapping from zero-indexed document IDs to a bag of words
		(mapping from word to the number of times it appears)
		2. A mapping from words to the number of documents it appears in
		3. A mapping from words to unique, zero-indexed integer IDs
		4.  A mapping from document IDs to labels (politics, entertainment, tech, etc)
	"""
	documents = {} # Mapping from document IDs to a bag of words
	document_word_counts = Counter() # Mapping from words to number of documents it appears in
	word_ids = {} # Mapping from words to unique integer IDs
	labels = {} # Mapping from document IDs to labels

	doc_id = 0
	words = set()

	for filename in os.listdir(data_dir):
		filepath = os.path.join(data_dir, filename)

		with open(filepath, 'r', errors='ignore') as f:
			doc = process_document(f.read())
			# TODO: update documents, document_word_counts, word_ids, and labels
			documents[doc_id] = Counter(doc)
			document_word_counts.update(set(doc))
			words.update(doc)
			labels[doc_id] = filename[0]
			doc_id += 1

	word_index = 0
	for w in words:
		word_ids[word_index] = w
		word_index += 1

	return documents, document_word_counts, word_ids, labels


def lsa(documents, document_word_counts, word_ids, num_topics=100, topics_per_document=3):
	"""
	Implements the LSA (Latent Semantic Analysis) algorithm
	to perform topic modeling.

	Args:
		documents: A mapping from zero-indexed document IDs to a bag of words
		document_word_counts: A mapping from words to the number of documents it appears in
		word_ids: A mapping from words to unique, zero-indexed integer IDs
	Returns:
		A dictionary that maps document IDs to a list of topics.
	"""
	# TODO: find the number of documents and words
	num_documents = len(documents)
	num_words = len(word_ids)

	tf_idf = np.zeros([num_documents, num_words])

	# TODO: calculate the values in tf_idf and store them in the appropriate position in the tf_idf matrix
	for i in range(0, num_documents):
		for j in range(0, num_words):
			tf = documents[i][word_ids[j]] / len(documents[i])
			idf = np.log(num_documents / document_word_counts[word_ids[j]])
			tf_idf[i][j] = tf*idf

	# TODO: use matrix factorization to transform tf_idf matrix into a document topic matrix, where
	# rows represent documents and columns represent topics (refer to MF lab)
	document_topic_matrix = TruncatedSVD(random_state=0,n_components=num_topics).fit_transform(tf_idf)

	# TODO: return a dictionary that maps document IDs to a list of each one's top "topics_per_document" topics
	topics_per_document = {}
	for i in range(0, num_documents):
		row = document_topic_matrix[i].argsort()[-3:]
		topics_per_document[i] = list(row)
	return topics_per_document


def classify_documents(topics, labels):
	"""
	Classifies documents based on their topics.

	Args:
		topics: a dictionary that maps document IDs to topics.
		labels: labels for each of the test files
	Returns:
		The score of each of the classifiers on the test data.
	"""

	def classify(classifier):
		"""
		Trains a classifier and tests its performance.

		NOTE: since this is an inner function within
		classify_documents, this function will have access
		to the variables within the scope of classify_documents,
		including the train and test data, so we don't need to pass
		them in as arguments to this function.

		Args:
			classifier: an sklearn classifier
		Returns:
			The score of the classifier on the test data.
		"""
		# TODO: fit the classifier on X_train and y_train
		# and return the score on X_test and y_test
		classifier.fit(X_train, y_train)
		return classifier.score(X_test, y_test)

	# TODO: create X and Y from topics and labels
	X = np.vstack(list(topics.values()))
	y = list(labels.values())

	# TODO: use label_encoder to transform y
	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(y)

	# TODO: modify the call to train_test_split to use
	# 90% of the data for training and 10% for testing.
	# Make sure to also shuffle and set a random state of 0!
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.1,
		random_state=0
	)

	# TODO: create a KNeighborsClassifier that uses 3 neighbors to classify
	knn = KNeighborsClassifier(n_neighbors=3)
	knn_score = classify(knn)

	# TODO: create a DecisionTreeClassifier with random_state=0
	decision_tree = DecisionTreeClassifier(random_state=0)
	decision_tree_score = classify(decision_tree)

	# TODO: create an SVC with random_state=0
	svm = SVC(random_state=0)
	svm_score = classify(svm)

	# TODO: create an MLPClassifier with random_state=0
	mlp = MLPClassifier(random_state=0)
	mlp_score = classify(mlp)

	return knn_score, decision_tree_score, svm_score, mlp_score


def cluster_documents(topics, num_clusters=4):
	"""
	Clusters documents based on their topics.

	Args:
		document_topics: a dictionary that maps document IDs to topics.
	Returns:
		1. the predicted clusters for each document. This will be a list
		in which the first element is the cluster index for the first document
		and so on.
		2. the centroid for each cluster.
	"""
	k_means = KMeans(n_clusters=num_clusters, random_state=0)

	# TODO: Use k_means to cluster the documents and return the clusters and centers
	return k_means.fit_predict(np.vstack(list(topics.values()))), k_means.cluster_centers_


def plot_clusters(document_topics, clusters, centers):
	"""
	Uses matplotlib to plot the clusters of documents

	Args:
		document_topics: a dictionary that maps document IDs to topics.
		clusters: the predicted cluster for each document.
		centers: the coordinates of the center for each cluster.
	"""
	topics = np.array([x for x in document_topics.values()])

	ax = plt.figure().add_subplot(111, projection='3d')
	ax.scatter(topics[:, 0], topics[:, 1], topics[:, 2], c=clusters, alpha=0.3) # Plot the documents
	ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', alpha=1) # Plot the centers

	plt.tight_layout()
	plt.show()


def main(data_dir):
	"""
	This runs the program!

	Args:
		data_dir: the path to the money diaries
	"""
	# Read in the data
	documents, document_word_counts, word_ids, labels = read_data(data_dir)

	# Perform LSA
	topics = lsa(documents, document_word_counts, word_ids)

	# Classify the data
	knn_score, decision_tree_score, svm_score, mlp_score = classify_documents(topics, labels)

	print('\n===== CLASSIFIER PERFORMANCE =====')
	print('K-Nearest Neighbors Accuracy: %.3f' % knn_score)
	print('Decision Tree Accuracy: %.3f' % decision_tree_score)
	print('SVM Accuracy: %.3f' % svm_score)
	print('Multi-Layer Perceptron Accuracy: %.3f' % mlp_score)
	print('\n')

	# Cluster the data
	clusters, centers = cluster_documents(topics)
	plot_clusters(topics, clusters, centers)

# Run using 'python nlp.py' or 'python nlp.py <PATH_TO_BBC_DIRECTORY>'
# to manually specify the path to the data.
# This may take a little bit of time (~30-60 seconds) to run.
if __name__ == '__main__':
	data_dir = 'data' if len(sys.argv) == 1 else sys.argv[1]
	main(data_dir)

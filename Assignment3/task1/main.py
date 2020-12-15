import os
import numpy as np
import sklearn
import re
import pandas as pd
import csv
import spacy
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def createPredDir(path):
	if not os.path.isdir(path):
		os.mkdir(path)


def readData(path, train_file, test_file):
	f = open( os.path.join(path, train_file), 'r')
	train_data = []
	c = 1
	for line in f:
		t = line.split("\t")
		if c == 1:
			c += 1
			continue
		train_data.append([int(t[0]) , t[1], int(t[2].split('\n')[0])])
	f.close()

	f = open( os.path.join(path, test_file), 'r')
	test_data = []
	c = 1
	for line in f:
		t = line.split("\t")
		if c == 1:
			c += 1
			continue
		test_data.append([int(t[0]) ,t[1][:-1]])
	f.close()

	return np.array(train_data), np.array(test_data)


def preprocess(train, test):
	preprocessed_train, preprocessed_test = [], []

	for row in train:
		text = re.sub(r'[^\w\s]', '', row[1]) 
		preprocessed_train.append([row[0], text.lower(), row[2]])

	for row in test:
		text = re.sub(r'[^\w\s]', '', row[1]) 
		preprocessed_test.append([row[0], text.lower()])

	return np.array(preprocessed_train), np.array(preprocessed_test)



def writePred(array, path, filename):
	'''
	Writes the prediction in a file
	'''
	fields = ["id", "hateful"]
	rows = [ [ elem[0][0], elem[1] ] for elem in array]
	with open(os.path.join(path, filename), 'w') as csvfile:
		csvwriter = csv.writer(csvfile) 
		csvwriter.writerow(fields)
		csvwriter.writerows(rows)




def getTFIDF(document):
	'''
	Finds tfidf vectors for document
	'''
	tf = TfidfVectorizer(input='content', encoding='latin1', analyzer='word',
					 min_df = 5, max_df=0.8, use_idf=True, smooth_idf=True)
	tfidf =  tf.fit_transform(document)
	return tfidf



def RandomForest(train, test, path_to_save):
	'''
	Fits Random Forest classifier with 80 estimators and writes the prediction of test set
	'''
	tfidf = getTFIDF( list(zip(*train))[1] + list(zip(*test))[1] )
	tfidf_train = tfidf[:train.shape[0]]
	tfidf_test = tfidf[train.shape[0]:] 

	forest = RandomForestClassifier(n_estimators=80, max_depth=15, random_state=0)
	forest.fit( tfidf_train, list(zip(*train))[2] )

	pred = forest.predict(tfidf_test)
	writePred(list(zip(test, pred)), path=path_to_save, filename='RF.csv')




def getWord2Vec(document):
	'''
	Find Word2Vec for a list of words
	'''
	nlp = spacy.load("en_core_web_md")

	vectors = []
	for row in document:
		tokens = nlp(str(row))
		if len(tokens) == 0:
			emb = [0 for _ in range(300)]
		else:
			emb = np.mean( [token.vector for token in tokens], axis=0 )
		vectors.append(emb)

	return np.array(vectors)	



def SVM(train, test, path_to_save):
	'''
	Runs SVM with RBF kernel and outputs the predictions on test data
	'''
	word2vec_train = getWord2Vec( list(zip(*train))[1] )
	word2vec_test = getWord2Vec( list(zip(*test))[1] )

	svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto'))
	svm.fit( word2vec_train, list(zip(*train))[2] )

	pred = svm.predict(word2vec_test)
	writePred(list(zip(test, pred)), path=path_to_save, filename='SVM.csv')	




def FastText(train, test, path_to_save):
	'''
	Runs FastText supervised model and writes the predictions
	'''
	directory = os.path.join(os.getcwd(), 'fasttext_train')
	if not os.path.isdir(directory):
		os.mkdir(directory)
	
	filename = os.path.join(directory, 'file.train')
	f = open( filename, 'w' )

	# Store the training data in a file
	for row in train:
		row[1] = row[1].replace("\n", " ")
		row[1] = row[1].replace("\'", " ")
		f.write('__label__{} {}\n'.format(row[2], row[1]))

	model =  fasttext.train_supervised(input=filename, dim=300, epoch=15, verbose=0)

	# Find the predictions on test data
	predictions = []
	for row in test:
		row[1] = row[1].replace("\n", " ")
		row[1] = row[1].replace("\'", " ")
		pred = model.predict(row[1])
		predictions.append(([row[0], row[1]], int(pred[0][0][-1]) ))

	writePred(predictions, path=path_to_save, filename='FT.csv')



def main():
	DATA_PATH = os.path.join(os.getcwd(), '../data')
	PREDICTION_PATH = os.path.join(os.getcwd(), '../predictions')
	train_file_name = 'train.tsv'
	test_file_name = 'test.tsv'
	
	createPredDir(PREDICTION_PATH)

	# Read and preprocess data
	train, test = readData(DATA_PATH, train_file_name, test_file_name)
	train, test = preprocess(train, test)

	# Run Several models
	RandomForest(train, test, path_to_save=PREDICTION_PATH)
	SVM(train, test, path_to_save=PREDICTION_PATH)
	FastText(train, test, path_to_save=PREDICTION_PATH)	


if __name__ == "__main__":
	main()
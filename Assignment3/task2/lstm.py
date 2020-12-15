from data_handler import get_data
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Bidirectional, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold
from keras.utils import np_utils
import codecs
import operator
import gensim, sklearn
from string import punctuation
from collections import defaultdict
from batch_gen import batch_gen
import sys
import csv
import os
from nltk import tokenize as tokenize_nltk
from my_tokenizer import glove_tokenize


### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
train_tweets, test_tweets = [], []


EMBEDDING_DIM = None
GLOVE_MODEL_FILE = None
SEED = 42
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = None
OPTIMIZER = None
KERNEL = None
TOKENIZER = None
MAX_SEQUENCE_LENGTH = None
INITIALIZE_WEIGHTS_WITH = None
LEARN_EMBEDDINGS = None
EPOCHS = None
BATCH_SIZE = None
SCALE_LOSS_FUN = None

word2vec_model = None


def get_embedding(word):
	try:
		return word2vec_model[word]
	except:
		print('Encoding not found: %s' %(word))
		return np.zeros(EMBEDDING_DIM)


def get_embedding_weights():
	embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
	n = 0
	for k, v in vocab.items():
		try:
			embedding[v] = word2vec_model[k]
		except:
			n += 1
			pass
	return embedding


def writePred(array, path, filename):
	fields = ["id", "hateful"]
	rows = [ [ elem[0][0], elem[1] ] for elem in array]
	with open(os.path.join(path, filename), 'w') as csvfile:
		csvwriter = csv.writer(csvfile) 
		csvwriter.writerow(fields)
		csvwriter.writerows(rows)


def loadGloveModel(File):
	# Loads the Glove Embedding file
	f = open(File,'r')
	gloveModel = {}
	for line in f:
		splitLines = line.split()
		word = splitLines[0]
		wordEmbedding = np.array([float(value) for value in splitLines[1:]])
		gloveModel[word] = wordEmbedding
	return gloveModel


def select_tweets():
	# selects the tweets as in mean_glove_embedding method
	train_tweets, test_tweets = get_data()
	tweet_return = []
	for tweet in train_tweets:
		_emb = 0
		words = TOKENIZER(tweet['text'].lower())
		for w in words:
			if w in word2vec_model:  # Check if embedding there in GLove model
				_emb += 1
		if _emb:
			tweet_return.append(tweet)

	return tweet_return, test_tweets


def gen_vocab():
	# Create a dictionary of all the words (vocabulary)
	vocab_index = 1
	for tweet in train_tweets:
		text = TOKENIZER(tweet['text'].lower())
		text = ' '.join([c for c in text if c not in punctuation])
		words = text.split()
		words = [word for word in words if word not in STOPWORDS]

		for word in words:
			if word not in vocab:
				vocab[word] = vocab_index
				reverse_vocab[vocab_index] = word       # generate reverse vocab as well
				vocab_index += 1
			freq[word] += 1

	for tweet in test_tweets:
		text = TOKENIZER(tweet['text'].lower())
		text = ' '.join([c for c in text if c not in punctuation])
		words = text.split()
		words = [word for word in words if word not in STOPWORDS]

		for word in words:
			if word not in vocab:
				vocab[word] = vocab_index
				reverse_vocab[vocab_index] = word
				vocab_index += 1
			freq[word] += 1

	vocab['UNK'] = len(vocab) + 1
	reverse_vocab[len(vocab)] = 'UNK'


def filter_vocab(k):
	global freq, vocab
	freq_sorted = sorted(freq.items(), key=operator.itemgetter(1))
	tokens = freq_sorted[:k]
	vocab = dict(zip(tokens, range(1, len(tokens) + 1)))
	vocab['UNK'] = len(vocab) + 1


def gen_train_sequence():
	# Generate training sequence to give input to model
	X, y = [], []
	for tweet in train_tweets:
		text = TOKENIZER(tweet['text'].lower())
		text = ' '.join([c for c in text if c not in punctuation])
		words = text.split()
		words = [word for word in words if word not in STOPWORDS]
		seq, _emb = [], []
		for word in words:
			seq.append(vocab.get(word, vocab['UNK']))
		X.append(seq)
		y.append(tweet['label'])
	return X, y


def gen_test_sequence():
	# Generate testing sequence for predicting from model
	X = []
	for tweet in test_tweets:
		text = TOKENIZER(tweet['text'].lower())
		text = ' '.join([c for c in text if c not in punctuation])
		words = text.split()
		words = [word for word in words if word not in STOPWORDS]
		seq, _emb = [], []
		for word in words:
			seq.append(vocab.get(word, vocab['UNK']))
		X.append(seq)
	return X


def shuffle_weights(model):
	weights = model.get_weights()
	weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
	model.set_weights(weights)


def lstm_model(sequence_length, embedding_dim):
	# Defines LSTM model
	model = Sequential()
	model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
	model.add(Dropout(0.25))
	model.add(LSTM(50))
	model.add(Dropout(0.5))
	model.add(Dense(2))
	model.add(Activation('softmax'))
	model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])
	print(model.summary())
	return model


def train_LSTM(X_training, y_training, X_testing, model, inp_dim, weights, epochs, batch_size):
	# Trains lstm model by performing K-Fold Cross Validation
	prec_macro, recall_macro, f1_macro = 0., 0., 0.
	prec_micro, recall_micro, f1_micro = 0., 0., 0.
	sentence_len = X_training.shape[1]

	cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)

	for train_index, val_index in cv_object.split(X_training):
		if INITIALIZE_WEIGHTS_WITH == "glove":
			model.layers[0].set_weights([weights])
		elif INITIALIZE_WEIGHTS_WITH == "random":
			shuffle_weights(model)
		else:
			print("ERROR!")
			return
		X_train, y_train = X_training[train_index], y_training[train_index]
		X_val, y_val = X_training[val_index], y_training[val_index]
		y_train = y_train.reshape((len(y_train), 1))
		X_temp = np.hstack((X_train, y_train))

		for epoch in range(epochs):
			loss, acc = -1, -1

			for X_batch in batch_gen(X_temp, batch_size):
				x = X_batch[:, :sentence_len]
				y_temp = X_batch[:, sentence_len]

				class_weights = None
				if SCALE_LOSS_FUN:
					class_weights = {}
					class_weights[0] = np.where(y_temp == 0)[0].shape[0]/float(len(y_temp))
					class_weights[1] = np.where(y_temp == 1)[0].shape[0]/float(len(y_temp))

				y_temp = np_utils.to_categorical(y_temp)
				loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)

			print("Epoch: {}\tLoss: {}\tAccuracy: {}".format(epoch, round(loss,5), round(acc,5)))

		y_pred = model.predict_on_batch(X_val)
		y_pred = np.argmax(y_pred, axis=1)
		print(classification_report(y_val, y_pred))
		print("")
		prec_macro += precision_score(y_val, y_pred, average='weighted')
		prec_micro += precision_score(y_val, y_pred, average='micro')
		recall_macro += recall_score(y_val, y_pred, average='weighted')
		recall_micro += recall_score(y_val, y_pred, average='micro')
		f1_macro += f1_score(y_val, y_pred, average='weighted')
		f1_micro += f1_score(y_val, y_pred, average='micro')

	X_id = [(x["id"], x["text"]) for x in test_tweets]
	y_pred = model.predict_on_batch(X_testing)
	y_pred = np.argmax(y_pred, axis=1)
	writePred(list(zip(X_id, y_pred)), path='../predictions/', filename='T2.csv')

	print("\n\n## FINAL RESULTS ##")
	print("MACRO Results: ")
	print("Avg. Precision: %f" %(prec_macro/NO_OF_FOLDS))
	print("Avg. Recall: %f" %(recall_macro/NO_OF_FOLDS))
	print("Avg. Macro-F1: %f" %(f1_macro/NO_OF_FOLDS))

	print("\nMICRO results: ")
	print("Avg. Precision: %f" %(prec_micro/NO_OF_FOLDS))
	print("Avg. Recall: %f" %(recall_micro/NO_OF_FOLDS))
	print("Avg. Micro-F1: %f" %(f1_micro/NO_OF_FOLDS))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='LSTM based models for twitter Hate speech detection')
	parser.add_argument('-f', '--embeddingfile', required=True)
	parser.add_argument('-d', '--dimension', required=True)
	parser.add_argument('--tokenizer', choices=['glove', 'nltk'], required=True)
	parser.add_argument('--loss', default=LOSS_FUN, required=True)
	parser.add_argument('--optimizer', default=OPTIMIZER, required=True)
	parser.add_argument('--epochs', default=10, required=True)
	parser.add_argument('--batch-size', default=512, required=True)
	parser.add_argument('-s', '--seed', default=SEED)
	parser.add_argument('--folds', default=NO_OF_FOLDS)
	parser.add_argument('--kernel', default=KERNEL)
	parser.add_argument('--class_weight')
	parser.add_argument('--initialize-weights', choices=['random', 'glove'], required=True)
	parser.add_argument('--learn-embeddings', action='store_true', default=False)
	parser.add_argument('--scale-loss-function', action='store_true', default=False)

	args = parser.parse_args()
	GLOVE_MODEL_FILE = args.embeddingfile
	EMBEDDING_DIM = int(args.dimension)
	SEED = int(args.seed)
	NO_OF_FOLDS = int(args.folds)
	CLASS_WEIGHT = args.class_weight
	LOSS_FUN = args.loss
	OPTIMIZER = args.optimizer
	KERNEL = args.kernel
	if args.tokenizer == "glove":
		TOKENIZER = glove_tokenize
	elif args.tokenizer == "nltk":
		TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
	INITIALIZE_WEIGHTS_WITH = args.initialize_weights    
	LEARN_EMBEDDINGS = args.learn_embeddings
	EPOCHS = int(args.epochs)
	BATCH_SIZE = int(args.batch_size)
	SCALE_LOSS_FUN = args.scale_loss_function

	np.random.seed(SEED)
	
	print("\n")
	print("Embedding Dimension: {}".format(EMBEDDING_DIM))
	print("Model: LSTM")
	print("Num Epochs: {}".format(EPOCHS))
	print("Batch Size: {}".format(BATCH_SIZE))
	print("Num Folds for K-Fold Cross Validation: {}".format(NO_OF_FOLDS))
	print("")

	print("Loading Glove Embedding file...")
	word2vec_model = loadGloveModel(GLOVE_MODEL_FILE)
	print("Glove Embedding file loaded")

	train_tweets, test_tweets = select_tweets()
	print('Number of Training Tweets selected:', len(train_tweets))
	gen_vocab()
	X_train, y_train = gen_train_sequence()
	X_test = gen_test_sequence()

	MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X_train))
	data_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
	data_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
	y_train = np.array(y_train)
	data_train, y_train = sklearn.utils.shuffle(data_train, y_train)
	W = get_embedding_weights()


	print("\nDefining Model...")
	model = lstm_model(data_train.shape[1], EMBEDDING_DIM)

	print("\nTraining...")
	train_LSTM(data_train, y_train, data_test, model, EMBEDDING_DIM, W, epochs=EPOCHS, batch_size=BATCH_SIZE)

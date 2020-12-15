import re
import pandas as pd

def get_data():
	f = open("../data/train.tsv", 'r')
	train = []
	c = 1
	for line in f:
		t = line.split("\t")
		if c == 1:
			c += 1
			continue
		train.append([int(t[0]) , t[1], int(t[2].split('\n')[0])])
	f.close()

	f = open("../data/test.tsv", 'r')
	test = []
	c = 1
	for line in f:
		t = line.split("\t")
		if c == 1:
			c += 1
			continue
		test.append([int(t[0]) ,t[1][:-1]])
	f.close()

	preprocessed_train, preprocessed_test = [], []

	for row in train:
		text = re.sub(r'[^\w\s]', '', row[1]) 
		preprocessed_train.append({
			"id": row[0],
			"text": text.lower(),
			"label": row[2]
			})

	for row in test:
		text = re.sub(r'[^\w\s]', '', row[1]) 
		preprocessed_test.append({
			"id": row[0],
			"text": text.lower()
			})

	return preprocessed_train, preprocessed_test

import nltk
import os
import re
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

FOLDER = './processed_acl/'
lemmatizer = nltk.stem.WordNetLemmatizer()

def read_file(folder):
	os.system('mkdir tf_record')
	for sub_folder in os.listdir(folder):
		os.system('mkdir ./tf_record/{}'.format(sub_folder))
		print('Folder : ',sub_folder)
		negative = open(FOLDER+sub_folder+'/'+'negative.review')
		positive = open(FOLDER+sub_folder+'/'+'positive.review')
		
		neg_content = negative.read()
		pos_content = positive.read()
		neg_content = neg_content.split('#label#:negative')
		pos_content = pos_content.split("#label#:positive")

		pos_d = tokenize(pos_content)
		neg_d = tokenize(neg_content)

		create_image(pos_d,neg_d,sub_folder)
		

def tokenize(data):
	text = ''
	for word in data:
		text += re.sub('[^a-zA-Z]', ' ', word)

	text = text.replace('  ',' ')
	text = text.split()
	lemmatized = []
	for word in tqdm(text):
		if word not in nltk.corpus.stopwords.words('english'):
			lemmatized.append(lemmatizer.lemmatize(word))

	lemmatized = ' '.join(lemmatized)
	sentences = nltk.word_tokenize(lemmatized)
	tokenizer = Tokenizer(oov_token='<OOV>')
	tokenizer.fit_on_texts(sentences)
	word_index = tokenizer.word_index
	sequence = tokenizer.texts_to_sequences(sentences)
	return sequence

def create_image(positive,negative,category,image_size=64,):
	max_len = min(len(positive),len(negative))

	pos_data = np.array(positive[:max_len]).reshape(-1,)
	neg_data = np.array(negative[:max_len]).reshape(-1,)

	embedding = tf.keras.layers.Embedding(20000,image_size)
	batches = int(np.floor(max_len/image_size))
	data = []
	label = []
	for img_batch in tqdm(range(batches)):
		img_pos = embedding(tf.constant(np.array(pos_data[img_batch*(image_size):(img_batch+1)*image_size])))
		img_neg = embedding(tf.constant(np.array(neg_data[img_batch*(image_size):(img_batch+1)*image_size])))
		data.append(img_pos)
		data.append(img_neg)
		label.append(1)
		label.append(0)

		if (img_batch+1) % 1000 == 0:
			img_list = tf.train.FloatList(value=np.array(data).reshape(-1,))
			label_list = tf.train.Int64List(
                value=np.array(label))

			image = tf.train.Feature(float_list=img_list)
			labels = tf.train.Feature(int64_list=label_list)

			fully = {'image': image,'labels': labels}
			full = tf.train.Features(feature=fully)
			example = tf.train.Example(features=full)

			with tf.io.TFRecordWriter('./tf_record/'+category+'/'+str(img_batch)+'.tfrecord') as writer:
				writer.write(example.SerializeToString())

			data = []
			label = []


read_file(FOLDER)

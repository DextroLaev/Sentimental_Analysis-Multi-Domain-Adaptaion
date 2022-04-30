import tensorflow as tf
import numpy as np
import os

base_path = './tf_record'

def _parse_function(example_proto):
	feature_desc = {
	'image':tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0,allow_missing=True),
	'labels':tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,allow_missing=True),
	}

	return tf.io.parse_single_example(example_proto,feature_desc)

class Datagenerator(tf.keras.utils.Sequence):
	def __init__(self,data,labels,shuffle=True,batch_size=1000):
		self.data = data
		self.labels = labels
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.indexes = np.arange(len(self.data))

	def __len__(self):
		return int(np.floor(len(self.data)/self.batch_size))

	def __getitem__(self,index):
		indexs = self.indexes[(index)*self.batch_size:(index+1)*self.batch_size]
		return self.data[indexs],self.labels[indexs]
	
	def on_epoch_end(self):
		np.random.shuffle(self.indexes)	

class Data:
	def __init__(self,data_path):
		self.objects =  [i for i in os.listdir(data_path)]
		self.base_path = base_path
		self.batch_size = 1000

	def task(self,index):
		if 0 <= index <= len(self.objects):
			index -= 1
		else:
			return 'No dataset present on this index'

		data = []
		labels = []
		files = [self.base_path+'/'+self.objects[index]+'/'+ i for i in os.listdir(self.base_path+'/'+self.objects[index])]

		for file in files:
			raw = tf.data.TFRecordDataset(file).map(_parse_function)
			for parsed_record in raw.take(5):
				img = np.array(parsed_record['image']).reshape(2000,64,64)
				label = np.array(parsed_record['labels']).reshape(2000,1)
				data.append(img)
				labels.append(label)

		data = np.array(data).reshape(len(data)*2000,64,64,1)
		labels = np.array(labels).reshape(len(labels)*2000,1)

		return Datagenerator(data,labels)

# if __name__ == '__main__':
	# obj = Data('./tf_record')
	# n = obj.task(1)
	# print(len(n.__getitem__(1)))
import numpy as np
import tensorflow as tf
from read_tf_record import Data
from model_architect import lenet5
import matplotlib.pyplot as plt

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.accA=[]
        self.accB=[]
    
    def on_epoch_end(self,epoch,logs=None):
        self.accB.append(logs['accuracy'])
        self.accA.append(logs['val_accuracy'])
        if logs['accuracy']==1.0:
            self.model.stop_training=1  

    def on_train_end(self,logs=None):
        plt.plot(self.accB,label='accuracy')
        plt.plot(self.accA,label='val_accuracy')
        plt.legend()
        plt.show()           

class Continual_Learning_Neural_net:
    def __init__(self,lambda_val):
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.lambda_val = lambda_val
        self.model = self.model_arch()
        self.theta = []
        self.theta_star = []

    def model_arch(self):
        model = lenet5()
        model.compile(loss=self.loss_fn,optimizer='adam',metrics=['accuracy'])
        return model

    def update_weights(self):
        self.theta = self.model.weights
        self.theta_star = self.model.get_weights()    

    def train(self,train_gen,epochs=10):
        self.model.fit(train_gen,epochs=epochs) 
        self.update_weights()
    
    def fisher_matrix(self,data_gen):
        self.fisher = [tf.zeros(v.get_shape().as_list()) for v in self.model.weights]
       
        # extracting data from data_gen
        data_set = []
        for gen in data_gen:
        	gen_len = gen.__len__()
        	partial_data = []

        	for d in range(gen_len):
        		partial_data.append(gen.__getitem__(d)[0])

        		data_set.append(np.array(partial_data))	

        count = 0
        for data in data_set:
            count += len(data[0])//20
            for i in range(len(data[0])//20):
                d = tf.reshape(data[0][i],(1,64,64))
                with tf.GradientTape() as tape:
                    tape.watch(self.model.weights)
                    probs = self.model(d)
                    y = tf.math.log(probs)

                grad = tape.gradient(y,[v for v in self.model.weights])

                for v in range(len(self.fisher)):
                    self.fisher[v] += tf.square(grad[v])

        for v in range(len(self.fisher)):
            self.fisher[v] /= count  
        return self.fisher    
    
    def custom_loss(self,y_true,y_pred):
        loss = self.loss_fn(y_true,y_pred)
        for v in range(len(self.theta)):
            loss += (self.lambda_val/2)*tf.reduce_sum(tf.multiply(self.fisher[v],tf.square(self.theta[v]-self.theta_star[v])))
        return loss        

    def train_with_ewc(self,train_gen,val_gen,epochs=10):
        self.fisher = self.fisher_matrix(val_gen) 

        self.model.compile(loss=self.custom_loss,optimizer='adam',metrics=['accuracy'])
        cb = CustomCallback()
        self.model.fit(train_gen,epochs=epochs,validation_data=(val_gen[-1]),callbacks=[cb])

        if len(val_gen) == 2:
            acc_C = self.model.evaluate(train_gen)[1]
            acc_B = self.model.evaluate(val_gen)[1]
            acc_A = self.model.evaluate(val_gen)[1]

            print('\nTask A:',acc_A)
            print('Task B:',acc_B)
            print('Task C:',acc_C)

        self.update_weights()    

if __name__ == '__main__':
	tasks = Data('./tf_record')
	task1 = tasks.task(1)
	task2 = tasks.task(2)
	task3 = tasks.task(3)
	task4 = tasks.task(4)

	nn = Continual_Learning_Neural_net(lambda_val=5)
	nn.train(task1,epochs=20)

	print('fisher matrix')
	nn.train_with_ewc(task2,[task1],epochs=20)
	nn.train_with_ewc(task3,[task2,task1],epochs=20)

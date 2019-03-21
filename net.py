import keras.layers as L
import keras.models as M
from keras import metrics
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np,pdb
from embed import *
from keras import backend as K

## these metric functions taken from elsewhere ##

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
	"""Precision metric.
	Only computes a batch-wise average of precision.
	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def RNN1(): 
	# for the translations
	inputs = L.Input(shape=(train_pron_x.shape[1],29))
	layer = L.LSTM(n_neurons1,return_sequences=True)(inputs)
	layer = L.Dense(n_neurons1,name='FC2')(layer)
	layer = L.Activation('relu')(layer)
	layer = L.LSTM(n_neurons2,return_sequences=True)(layer)
	layer = L.Dropout(dropout_rate)(layer)
	layer = L.Dense(5,name='out_layer')(layer)
	layer = L.Activation('softmax')(layer)
	model = M.Model(inputs=inputs,outputs=layer)
	return model

def RNN2():
	inputs = L.Input(shape=(train_phon_x.shape[1],30))
	layer = L.LSTM(n_neurons1,return_sequences=True)(inputs)
	layer = L.Dense(n_neurons2,name='FC1')(layer)
	layer = L.Activation('relu')(layer)
	layer = L.LSTM(n_neurons1,return_sequences=True)(layer)
	layer = L.Dropout(dropout_rate)(layer)
	layer = L.Dense(5,name='out_layer')(layer)
	layer = L.Activation('softmax')(layer)
	model = M.Model(inputs=inputs,outputs=layer)
	return model

batch_size = 12
epochs = 25
n_neurons1 = 128
n_neurons2 = 256
dropout_rate = 0.5

model=RNN1()
model.compile(loss='categorical_crossentropy',optimizer='adam',
	metrics=[precision,recall,f1])
model.fit(train_pron_x,to_categorical(train_pron_y[:,:,np.newaxis]),
	batch_size=batch_size,epochs=epochs,validation_split=0.2,
	callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
model.save('{}_{}_{}_{}_{}_lstm_pron.h5'.format(
	batch_size,epochs,n_neurons1,n_neurons2,dropout_rate))

batch_size = 5
epochs = 25
n_neurons1 = 128
n_neurons2 = 256
dropout_rate = 0.5

model2 = RNN2()
model2.compile(loss='categorical_crossentropy',optimizer='adam',
	metrics=[precision,recall,f1])
model2.fit(train_phon_x,to_categorical(train_phon_y[:,:,np.newaxis]),
	batch_size=batch_size,epochs=epochs,validation_split=0.2,
	callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.00001)])
model2.save('{}_{}_{}_{}_{}_lstm_phon.h5'.format(
	batch_size,epochs,n_neurons1,n_neurons2,dropout_rate))

a = np.array([np.pad(a,((0,63-a.shape[0]),(0,0)),'constant') for a in test_pron_x])
b = np.array([np.pad(a,(0,63-a.shape[0]),'constant') for a in test_pron_y])
c = np.array([np.pad(a,((0,73-a.shape[0]),(0,0)),'constant') for a in test_phon_x])
d = np.array([np.pad(a,(0,73-a.shape[0]),'constant') for a in test_phon_y])

score1 = model.evaluate(a,to_categorical(b[:,:,np.newaxis]),verbose=1)
score2 = model2.evaluate(c,to_categorical(d[:,:,np.newaxis]),verbose=1)

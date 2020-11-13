import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence,text
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D
import numpy as np


data_set = pd.read_csv('Dataset.csv', header = None)

data_set.columns = ['Text', 'Category']
train_set = data_set.sample(frac=0.8)
data_set.drop(train_set.index,axis=0,inplace=True)
valid_set = data_set.sample(frac=0.5)
data_set.drop(valid_set.index,axis=0,inplace=True)
test_set = data_set

CLASSES= {'CPU_Utilization':0,'Password_Reset':1,'Memory_Utilization':2}
top_tokens=20000
max_len=50
filters=64
dropout_rate=0.2
embedding_dimension=200
kernel_size=3
pool_size=3

def data_map(df):
    return list(df['Text']),np.array(df['Category'].map(CLASSES))

train_text,train_labels = data_map(train_set)
valid_text,valid_labels=data_map(valid_set)
test_text,test_labels=data_map(test_set)

def embedding_matrix_conv(word_index, embedding_file_path, embedding_dimension):
    embedding_matrix_comb = {}
    with open(embedding_file_path,'r') as embed_file:
        for token_entry in embed_file:
            values = token_entry.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix_comb[word] = coefs
    num_words = min(len(word_index) + 1, top_tokens)
    embedding_matrix = np.zeros((num_words, embedding_dimension))
    for word, word_position in word_index.items():
        if word_position >= top_tokens:
            continue
        embedding_vector = embedding_matrix_comb.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_position] = embedding_vector
    return embedding_matrix

tokenizer=text.Tokenizer (num_words=top_tokens)
tokenizer.fit_on_texts(train_text)
word_index=tokenizer.word_index
embedding_file_path = 'glove.6B.200d.txt'


def create_model():
  model = models.Sequential()
  features = min(len(word_index) + 1, top_tokens)
  model.add(Embedding(input_dim=features,
                output_dim=embedding_dimension,
                input_length=max_len,
                weights=[embedding_matrix_conv(word_index,
                                embedding_file_path, embedding_dimension)],trainable=True))
  model.add(Dropout(rate=dropout_rate))
  model.add(Conv1D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                bias_initializer='he_normal',
                padding='same'))
  model.add(MaxPooling1D(pool_size=pool_size))
  model.add(Conv1D(filters=filters * 2,
                kernel_size=kernel_size,
                activation='relu',
                bias_initializer='he_normal',
                padding='same'))
  model.add(GlobalAveragePooling1D())
  model.add(Dropout(rate=dropout_rate))
  model.add(Dense(len(CLASSES), activation='softmax'))
  optimizer = tf.keras.optimizers.Adam(lr=0.001)
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
  return model

train_process = tokenizer.texts_to_sequences(train_text)
train_process = sequence.pad_sequences(train_process, maxlen=max_len)

valid_process = tokenizer.texts_to_sequences(valid_text)
valid_process = sequence.pad_sequences(valid_process, maxlen=max_len)

test_process = tokenizer.texts_to_sequences(test_text)
test_process = sequence.pad_sequences(test_process, maxlen=max_len)

model = create_model()
model.summary()

checkpoint_path = "training_path/cp.ckpt"
checkpoint_directory = os.path.dirname(checkpoint_path)

callback_path = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

model.fit(train_process,
          train_labels,
          epochs=10,
          validation_data=(test_process,test_labels),
          callbacks=[callback_path])
model.save('model_saved/model')

input_dict={'embedding_9_input': valid_process[1:2]}
with open('sample_instance.json', 'w') as prediction_file:
    json.dump(input_dict, prediction_file)

def predictClass():
    try:
        content= ['User requested to Change Password as expired']
        result = {}
        pred_process = tokenizer.texts_to_sequences(content)
        pred_process = sequence.pad_sequences(pred_process, maxlen=max_len)
        new_model = tf.keras.models.load_model('model_saved/model')
        prediction = int(new_model.predict_classes(pred_process))

        for key, value in CLASSES.items():
            if value==prediction:
                category=key
                result["class"] = category
        result = {"results": result}
        result = json.dumps(result)
        return result
    except Exception as e:
        return e

if __name__=='__main__':
	predictClass()

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from flask import Flask, request
import json

app=Flask(__name__)

CLASSES = {'CPU_Utilization': 0, 'Password_Reset': 1, 'Memory_Utilization': 2}
top_tokens = 20000
max_len = 50
def return_data(df):
    return list(df['Text']), np.array(df['Category'].map(CLASSES))

data = pd.read_csv('Dataset.csv', header = None)
data.columns = ['Text', 'Category']
train = data.sample(frac=0.8)

train_text, train_labels = return_data(train)

tokenizer = text.Tokenizer(num_words=top_tokens)
tokenizer.fit_on_texts(train_text)

app=Flask(__name__)

@app.route ("/predictclass", methods=['POST'])
def predictClass():
    try:
        json_data = request.get_json(force=True)
        content= json_data["content"]
        text = list(content)
        result = {}

        pred_process = tokenizer.texts_to_sequences(text)
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
        return {"Error": str(e)}
if __name__ == "__main__" :
    app.run(port="5000")

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:18:21 2021

@author: Nagesh
"""

# Reference: https://github.com/krishnaik06/Heroku-Demo/blob/master/app.py
#%%

import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import *
from flask import Flask, request, jsonify, render_template
import requests
import os

#%%

app = Flask(__name__)

#%%

if os.path.isexists("glove_vectors"):
    # Reference: https://www.tutorialspoint.com/downloading-files-from-web-using-python
    
    url = "https://github.com/nagi1995/sarcastic-comment-detection/blob/main/glove_vectors?raw=true"
    r = requests.get(url, allow_redirects = True)
    open('glove_vectors', 'wb').write(r.content)

#%%
t = pickle.load(open("tokenizer.pkl", 'rb'))
vocab_size = len(t.word_index) + 1

glove_model = pickle.load(open("glove_vectors", "rb"))
glove_words = set(glove_model.keys())

#%%
max_length = 25
embedding_matrix = np.zeros((vocab_size, 300)) # vector len of each word is 300

for word, i in t.word_index.items():
  if word in glove_words:
    vec = glove_model[word]
    embedding_matrix[i] = vec

#%%
tf.keras.backend.clear_session()
inp = Input(shape = (max_length, ), name = "input")
embedding = Embedding(input_dim = vocab_size, 
                      output_dim = 300, # glove vector size
                      weights = [embedding_matrix], 
                      trainable = False)(inp)
lstm = LSTM(32)(embedding)
flatten = Flatten()(lstm)
dense = Dense(16, activation = None, 
              kernel_initializer = "he_uniform")(flatten)
dropout = Dropout(.25)(dense)
activation = Activation("relu")(dropout)
output = Dense(2, activation = "softmax", name = "output")(activation)
model = Model(inputs = inp, outputs = output)
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.load_weights("weights.30-0.9999.hdf5")

#%%



def decontracted(phrase):
    # Reference: https://stackoverflow.com/a/47091490/6645883
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    #phrase = re.sub(r"[^A-Za-z0-9 ]+", "", phrase)
    return phrase.lower()


@app.route("/")
def home():
    print("\n\n#############")
    print("prediction_text")
    print("#############\n\n")
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == "POST":
        prediction_text = []
        try:
            text = request.form.get("text")
            text = decontracted(text)
            encoded_text = t.texts_to_sequences([text])
            padded_text = pad_sequences(encoded_text, maxlen = max_length, padding = "post", truncating = "post")
            prediction = model.predict(padded_text)
            prediction_text = "NOT SARCASTIC"
            if prediction[0][1] >= .5:
                prediction_text = "SARCASTIC"
            print("\n\n#############")
            print("prediction_text:", prediction_text)
            print("#############\n\n")
        except Exception  as e:
                print("error: ", e)
        return render_template('index.html', prediction_text = "The sms is {0}.".format(prediction_text))
    else:
        return render_template("index.html")

#%%
if __name__ == "__main__":
    app.run(debug = True)















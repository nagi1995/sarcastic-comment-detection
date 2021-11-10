# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:18:21 2021

@author: Nagesh
"""

# Reference: https://github.com/krishnaik06/Heroku-Demo/blob/master/app.py
#%%

import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify, render_template

#%%

app = Flask(__name__)

#%%


#%%
vectorizer = pickle.load(open("tfidf-vectorizer.pkl", 'rb'))
model = pickle.load(open("rf-model.pkl", 'rb'))


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
            x_test = vectorizer.transform([text])
            prediction = model.predict(x_test)
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















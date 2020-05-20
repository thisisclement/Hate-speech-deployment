"""
Predict hate speech class from sepal length and sepal width.
"""
from flask import Flask, jsonify, request
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.externals import joblib
from flask import render_template
import numpy as np
import spacy
import pickle

sp = spacy.load('en_core_web_sm')
import pandas as pd

app = Flask(__name__)

#load trained model from file
model = joblib.load('models/finalized_logreg.pkl')

@app.route('/', methods = ["GET", "POST"])
def predict_hatespeech_home():

    #init the output variables
    output = None
    prob = None
    labels = ['Non-hate Comment', 'Hate Comment']

    ## required functions ########
    def lemma(txt):
        text = sp(txt)
        #get all sentences
        sentence_lst = list(text.sents)
        lemma_words = []
        for sentence in sentence_lst:
            for word in sentence:
                lemma_words.append(word.lemma_)
        #return back as a string
        return ' '.join(lemma_words)
    ##############################

    if request.method == "POST":
        comment = str(request.form["comment"])
        lemm_comment = lemma(comment)
        cvec = pickle.load(open("models/cvec.pickle", "rb"))
        com_cvec = cvec.transform([lemm_comment])
        pred = model.predict(com_cvec)
        output = labels[pred[0]]
        prob = model.predict_proba(com_cvec)[0]
    return render_template("hatespeech.html", output = [output, prob])


@app.route('/predict-hatespeech', methods = ["GET", "POST"])
def predict_hatespeech():

    #init the output variables
    output = None
    prob = None
    labels = ['Non-hate', 'Hate']

    ## required functions ########
    def lemma(txt):
        text = sp(txt)
        #get all sentences
        sentence_lst = list(text.sents)
        lemma_words = []
        for sentence in sentence_lst:
            for word in sentence:
                lemma_words.append(word.lemma_)
        #return back as a string
        return ' '.join(lemma_words)
    ##############################

    if request.method == "POST":
        comment = str(request.form["comment"])
        lemm_comment = lemma(comment)
        cvec = pickle.load(open("../venv/cvec.pickle", "rb"))
        com_cvec = cvec.transform([lemm_comment])
        pred = model.predict(com_cvec)
        output = labels[pred[0]]
        prob = model.predict_proba(com_cvec)[0]
    return render_template("hatespeech.html", output = [output, prob])

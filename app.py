from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

import time
import os

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def isNaN(string):
    return string != string


def list_to_dict(words_list):
    return dict([(word, True) for word in words_list])


# Function to convert
def listToString(s):

    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele + " "

    # return string
    return str1

# ganti dari


def train():

    new_train_set = pd.read_excel('dataset/pretrained_dataset_train.xlsx')
    new_test_set = pd.read_excel('dataset/pretrained_dataset_test.xlsx')

    report = pd.read_csv('dataset/tabel.csv')
    report_precision_neg = report['precision'][0]  # negatif
    report_recall_neg = report['recall'][0]  # negatif
    report_f1_score_neg = report['f1-score'][0]  # negatif

    report_precision_pos = report['precision'][1]  # positif
    report_recall_pos = report['recall'][1]  # positif
    report_f1_score_pos = report['f1-score'][1]  # positif

    vektorisasi = TfidfVectorizer(min_df=5,
                                  max_df=0.8,
                                  sublinear_tf=True,
                                  use_idf=True)

    train_vektor = vektorisasi.fit_transform(new_train_set['tweet'])
    test_vektor = vektorisasi.transform(new_test_set['tweet'])

    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vektor, new_train_set['label'])
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vektor)
    t2 = time.time()

    tbpred = [0 if n == 'neg' else 1 for n in prediction_linear]
    y_validation = [0 if n == 'neg' else 1 for n in new_test_set['label']]
    conmat = np.array(confusion_matrix(y_validation, tbpred, labels=[1, 0]))
    confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
                             columns=['predicted_positive', 'predicted_negative'])
    print("Accuracy Score: {0:.2f}%".format(
        accuracy_score(y_validation, tbpred)*100))
    print("-"*80)
    print("Confusion Matrix\n")
    print(confusion)
    print("-"*80)
    print("Classification Report\n")
    print(classification_report(y_validation, tbpred))

    total_positif = 0
    total_negatif = 0
    total_label = 0

    for label in tbpred:
        if label == 1:
            total_positif += 1
        else:
            total_negatif += 1
        total_label += 1

    akurasi = accuracy_score(y_validation, tbpred)*100
    akurasi = round(akurasi, 2)

    sentimen = ''
    persentasi = 0
    if total_negatif > total_positif:
        sentimen = 'negatif'
        persentasi = (total_negatif/total_label)*100
    else:
        sentimen = 'positif'
        persentasi = (total_positif/total_label)*100

    persentasi = round(persentasi, 2)

    return (akurasi, report_precision_pos, report_recall_pos, report_f1_score_pos, report_precision_neg, report_recall_neg, report_f1_score_neg, total_positif, total_negatif, sentimen, persentasi)


@app.route('/', methods=['GET'])
def index():
    akurasi, report_precision_pos, report_recall_pos, report_f1_score_pos, report_precision_neg, report_recall_neg, report_f1_score_neg, total_positif, total_negatif, sentiment, persentasi = train()

    data = {"sentiment": sentiment, "accuracy": akurasi, "report_precision_pos": report_precision_pos, "report_recall_pos": report_recall_pos, "report_f1_score_pos": report_f1_score_pos,
            "report_precision_neg": report_precision_neg, "report_recall_neg": report_recall_neg, "report_f1_score_neg": report_f1_score_neg, "total_positif": total_positif, "total_negatif": total_negatif, "persentasi": persentasi}

    return jsonify(data), 200

@app.route('/dokumen', methods=['GET'])
def dokumen():
    dataset = pd.read_excel('dataset/pretrained_dataset_test.xlsx')

    positif_data = list()
    negatif_data = list()

    for (index, row) in dataset.iterrows():
        if row['label'] == 'pos':
            positif_data.append(row['tweet'])
        else:
            negatif_data.append(row['tweet'])

    data = {"positif": positif_data, "negatif": negatif_data}

    return jsonify(data), 200

@app.route('/test', methods=['GET'])
def test():

    message = "Berhasil terhubung ke Aplikasi"

    data = {'message': message}

    print(data)

    return jsonify(data), 200


if __name__ == "__main__":
    app.debug = True
    app.run()

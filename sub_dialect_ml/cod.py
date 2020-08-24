import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, classification_report
import pdb
import codecs
import csv
import pandas as pd
import re
import nltk


class Bag_of_words:

    def __init__(self): 
        self.words = []
        self.vocabulary_length = 0

    def build_vocabulary(self, data):
        for document in data:
            for word in document:
                if word not in self.words: 
                    self.words.append(word)

        self.vocabulary_length = len(self.words)
        self.words = np.array(self.words)
        
    def get_features(self, data):
        features = np.zeros((len(data), self.vocabulary_length))

        for document_idx, document in enumerate(data):
            for word in document:
                if word in self.words:
                    features[document_idx, np.where(self.words == word)[0][0]] += 1
        return features




rezultat= open("rezultat.txt","w+")


tl=open("train_labels.txt","r+")
vtl = [line for line in tl]
train_labels=[]
for i in range(len(vtl)):
    train_labels.append(vtl[i][7])


vl=open("validation_labels.txt","r+")
vvl = [line for line in vl]
validation_labels=[]
for i in range(len(vvl)):
    validation_labels.append(vvl[i][7])


ts=open("test_samples.txt","r+")
vts = [line for line in ts]
test_samples=[]
id=[]
for i in range(len(vts)):
    test_samples.append(vts[i][7:])
    id.append(vts[i][:6])

for i in range(len(test_samples)):
    test_samples[i]=test_samples[i][:-1]

for i in range(len(test_samples)):
    test_samples[i]=test_samples[i].split()


ts2=open("train_samples.txt","r+")
vts2 = [line for line in ts2]
train_samples=[]
for i in range(len(vts2)):
    train_samples.append(vts2[i][7:])

for i in range(len(train_samples)):
    train_samples[i]=train_samples[i][:-1]

for i in range(len(trainsamples)):
    train_samples[i]=train_samples[i].split()


vs=open("validation_samples.txt","r+")
vvs = [line for line in vs]
validation_samples=[]
for i in range(len(vvs)):
    validation_samples.append(vvs[i][7:])

for i in range(len(validation_samples)):
    validation_samples[i]=validation_samples[i][:-1]

for i in range(len(validation_samples)):
    validation_samples[i]=validation_samples[i].split()



bow_model = Bag_of_words()
bow_model.build_vocabulary(train_samples) 


def normalize_data(train_data, test_data, validation_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data) 
        scaled_validation_data = scaler.transform(validation_data) 
        return (scaled_train_data, scaled_test_data, scaled_validation_data)
    else:
        print("No scaling was performed. Raw data is returned.")
        return (train_data, test_data, scaled_validation_data)


train_features = bow_model.get_features(train_samples)
test_features = bow_model.get_features(test_samples)
validation_features = bow_model.get_features(validation_samples)


#print(train_features.shape)
#print(test_features.shape)
scaled_train_data, scaled_test_data, scaled_validation_data = normalize_data(train_features, test_features, validation_features, type='l2')


svm_model = svm.SVC(C=100, kernel='linear')
svm_model.fit(scaled_train_data, train_labels)
predicted_labels_svm = svm_model.predict(scaled_test_data) 
predicted_labels2_svm = svm_model.predict(scaled_validation_data)

print('f1 score', f1_score(np.asarray(validation_labels), predicted_labels2_svm, average='micro'))



labels = []
for i in range(len(predicted_labels_svm)):
    labels.append(predicted_labels_svm[i])
    

string = "id,label\n"
for i in range(len(predicted_labels_svm)):
    string += str(id[i]) + "," + str(labels[i]) + "\n"

rezultat.write(string)
rezultat.close()
tl.close()
vl.close()
ts.close()
ts2.close()
vs.close()


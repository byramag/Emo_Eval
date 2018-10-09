#        -- TODO --
# - x-fold cross validation
# - dimensionality reduction (fix)
# - alternative weighting
# - adding features
#   - pinpoint emojis
#   - differentiate between turns
#   - POS, NER
#   - find emotional words
# - other classifiers
#   - Naive Bayes baseline
#   - Random forest
#   - Neural Network
#   - different SVM kernel

# DEPENDENCIES: sklearn, pandas, pandas_ml, spacy

import re, sys
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from pandas_ml import ConfusionMatrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# import spacy
# import en_core_web_sm
# Spacy model setup
# nlp = en_core_web_sm.load()

t0 = time()

# Taking arg of file name, if no arg given, assumes train file is in the same directory
file_name = r"data/train.txt"
if len(sys.argv) > 1:
    file_name = sys.argv[1]

# reading training file into dataframe
print("Reading train file")
t = time()
instances = pd.read_csv(file_name, sep='\t', header=0)
print("Finished reading train file in %0.3fsec\n" % (time()-t))

# Separating the labels and strings into separate arrays & concatenating turns from bag of words
print("Separating text and labels")
t = time()
row_strings = []
labels = []
for index, instance in instances.iterrows():
    row_strings.append(instance['turn1'] + ' ' + instance['turn2'] + ' ' + instance['turn3'])
    labels.append(instance['label'])
print("Separating text and labels finished in %0.3fsec\n" % (time()-t))

# 60/40 split of training and test data
X_train, X_test, y_train, y_test = train_test_split(row_strings, labels, test_size=0.4, random_state=0)

# Creating bag of words feature vectors from training data
print("Creating training feature vectors")
t = time()
# countBoW = CountVectorizer() # add stopword removal or ngrams with: ngram_range=(1,2), stop_words='english'
tfidfBoW = TfidfVectorizer(ngram_range=(1,2))
X_train_mat = tfidfBoW.fit_transform(X_train)
print("Creating feature vectors finished in %0.3fsec\n" % (time()-t))

# # Dimensionality reduction of train matrix - THIS CURRENTLY GIVES BAD RESULTS
# print("Dimensionality reduction of train matrix")
# t = time()
# svd = TruncatedSVD(n_components=300)
# X_train_reduced = svd.fit_transform(X_train_mat)
# # pca = PCA(n_components=300)
# # X_train_reduced = pca.fit_transform(X_train_mat)
# print("Dimensionality reduction finished in %0.3fsec\n" % (time()-t))

# print (pd.DataFrame(data=X_train_mat.toarray(), columns=tfidfBoW.get_feature_names()))

# # Training Naive Bayes on training data # CURRENTLY CAUSES MEMORY ERROR
# print("Training Naive Bayes")
# t = time()
# nbClassifier = GaussianNB().fit(X_train_mat.toarray(), y_train)
# print("Training Naive Bayes finished in %0.3fsec\n" % (time()-t))

# Training Random Forest on training data
print("Training Random Forest")
t = time()
rfClassifier = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train_mat, y_train)
print("Training Random Forest finished in %0.3fsec\n" % (time()-t))

# Training linear kernel SVM on training data
print("Training SVM")
t = time()
svmClassifier = svm.SVC(kernel='linear', C=1).fit(X_train_mat, y_train)
print("Training SVM finished in %0.3fsec\n" % (time()-t))


# Creating bag of words feature vectors from test data
print("Vectorizing test features")
t = time()
X_test_mat = tfidfBoW.transform(X_test)
print("Vectorizing test features finished in %0.3fsec\n" % (time()-t))

# # Dimensionality reduction of test matrix - THIS CURRENTLY GIVES BAD RESULTS
# print("Dimensionality reduction of test matrix")
# t = time()
# X_test_reduced = svd.fit_transform(X_test_mat)
# # X_test_reduced = pca.fit_transform(X_test_mat)
# print("Dimensionality reduction finished in %0.3fsec\n" % (time()-t))

y_true = y_test

# # Evaluating NB metrics
# print("Predicting NB test instances\n")
# y_pred_nb = nbClassifier.predict(X_test_mat)
# confusion_matrix_nb = ConfusionMatrix(y_true, y_pred_nb)
# print("NB Confusion matrix:\n%s\n" % confusion_matrix_nb)

# accuracy_nb = accuracy_score(y_true, y_pred_nb)
# precision_nb = precision_score(y_true, y_pred_nb, average='macro')
# recall_nb = recall_score(y_true, y_pred_nb, average='macro')
# f1_nb = f1_score(y_true, y_pred_nb, average='macro')
# print("NB:\n\tAccuracy: %s\n\tPrecision: %s\n\tRecall: %s\n\tF1-Score: %s\n" % (accuracy_nb, precision_nb, recall_nb, f1_nb))

# Evaluating RF metrics
print("Predicting RF test instances\n")
y_pred_rf = rfClassifier.predict(X_test_mat)
confusion_matrix_rf = ConfusionMatrix(y_true, y_pred_rf)
print("RF Confusion matrix:\n%s\n" % confusion_matrix_rf)

accuracy_rf = accuracy_score(y_true, y_pred_rf)
precision_rf = precision_score(y_true, y_pred_rf, average='macro')
recall_rf = recall_score(y_true, y_pred_rf, average='macro')
f1_rf = f1_score(y_true, y_pred_rf, average='macro')
print("RF:\n\tAccuracy: %s\n\tPrecision: %s\n\tRecall: %s\n\tF1-Score: %s\n" % (accuracy_rf, precision_rf, recall_rf, f1_rf))
print("Micro averaged F1-Score: %s\n" % (f1_score(y_true, y_pred_rf, average='micro')))

# Evaluating SVM metrics
print("Predicting SVM test instances\n")
y_pred_svm = svmClassifier.predict(X_test_mat)
confusion_matrix = ConfusionMatrix(y_true, y_pred_svm)
print("SVM Confusion matrix:\n%s\n" % confusion_matrix)

accuracy_svm = accuracy_score(y_true, y_pred_svm)
precision_svm = precision_score(y_true, y_pred_svm, average='macro')
recall_svm = recall_score(y_true, y_pred_svm, average='macro')
f1_svm = f1_score(y_true, y_pred_svm, average='macro')
print("SVM:\n\tAccuracy: %s\n\tPrecision: %s\n\tRecall: %s\n\tF1-Score: %s\n" % (accuracy_svm, precision_svm, recall_svm, f1_svm))
print("Micro averaged F1-Score: %s\n" % (f1_score(y_true, y_pred_svm, average='micro')))

print("Total time for pipeline: %0.3fsec\n" % (time()-t0))
"""
Author: Abbie Byram
"""

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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# import spacy
# import en_core_web_sm
# Spacy model setup
# nlp = en_core_web_sm.load()


def preprocess(instances):
    """TODO: documentation
    """
    # Separating the labels and strings into separate arrays & concatenating turns from bag of words
    print("Preprocessing data")
    t = time()
    row_strings = []
    labels = []
    for index, instance in instances.iterrows():
        row_strings.append(instance['turn1'] + ' ' + instance['turn2'] + ' ' + instance['turn3'])
        labels.append(instance['label'])

    print("Finished preprocessing in %0.3fsec\n" % (time()-t))
    # 60/40 split of training and test data
    return train_test_split(row_strings, labels, test_size=0.4, random_state=0)


def feature_selection(x_train, x_test):
    # Creating bag of words feature vectors from training and test data
    print("Creating feature vectors")
    t = time()
    # countBoW = CountVectorizer() # add stopword removal or ngrams with: ngram_range=(1,2), stop_words='english'
    tfidfBoW = TfidfVectorizer(ngram_range=(1,2))
    x_train_mat = tfidfBoW.fit_transform(x_train)
    x_test_mat = tfidfBoW.transform(x_test)
    print("Creating feature vectors finished in %0.3fsec\n" % (time()-t))

    # # Dimensionality reduction - THIS CURRENTLY GIVES BAD RESULTS
    # print("Dimensionality reduction")
    # t = time()
    # svd = TruncatedSVD(n_components=300)
    # x_train_reduced = svd.fit_transform(x_train_mat)
    # x_test_reduced = svd.fit_transform(x_test_mat)
    # # pca = PCA(n_components=300)
    # # x_train_reduced = pca.fit_transform(x_train_mat)
    # # x_test_reduced = pca.fit_transform(x_test_mat)
    # print("Dimensionality reduction finished in %0.3fsec\n" % (time()-t))

    # print (pd.DataFrame(data=x_train_mat.toarray(), columns=tfidfBoW.get_feature_names()))

    return x_train_mat, x_test_mat


def train(x_train, x_test, y_train, y_test):
    # # Training Naive Bayes on training data # CURRENTLY CAUSES MEMORY ERROR
    # print("Training Naive Bayes")
    # t = time()
    # nbClassifier = GaussianNB().fit(x_train_mat.toarray(), y_train)
    # print("Training Naive Bayes finished in %0.3fsec\n" % (time()-t))

    print("Training Random Forest")
    t = time()
    rfClassifier = RandomForestClassifier(n_estimators=100, random_state=0).fit(x_train, y_train)
    print("Training Random Forest finished in %0.3fsec\n" % (time()-t))

    print("Training SVM")
    t = time()
    svmClassifier = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    print("Training SVM finished in %0.3fsec\n" % (time()-t))

    metrics = dict()

    # print("Predicting NB test instances\n")
    # metrics['NB'] = evaluate(nbClassifier.predict(x_test_mat), y_test)

    print("Predicting RF test instances\n")
    metrics['RF'] = evaluate(rfClassifier.predict(x_test), y_test)

    print("Predicting SVM test instances\n")
    metrics['SVM'] = evaluate(svmClassifier.predict(x_test), y_test)

    return metrics


def evaluate(pred, gold):
    """TODO: documentation
    """
    confusion_matrix = ConfusionMatrix(gold, pred)
    # print("Confusion matrix:\n%s\n" % confusion_matrix)

    accuracy = accuracy_score(gold, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(gold, pred, average='macro')
    # print("\tAccuracy: %s\n\tPrecision: %s\n\tRecall: %s\n\tF1-Score: %s\n" % (accuracy, precision, recall, f1))
    # print("Micro averaged F1-Score: %s" % (f1_score(gold, pred, average='micro')))
    # print("----------------------------------------------------------------")
    return {
        'acc': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main(train_file, sample_size=1, folds=5):
    # Reading training file into dataframe
    print("Reading train file")
    t = time()
    instances = pd.read_csv(file_name, sep='\t', header=0)
    print("Finished reading train file in %0.3fsec\n" % (time()-t))

    print("Sampling training data")
    t = time()
    instances = instances.sample(frac=sample_size)
    print("Finished sampling training data in %0.3fsec\n" % (time()-t))

    x_train, x_test, y_train, y_test = preprocess(instances)

    x_train, x_test = feature_selection(x_train, x_test)

    # Cross validation
    metrics = None
    for _ in range(folds):
        fold_results = train(x_train, x_test, y_train, y_test)
        if metrics is None:
            metrics = fold_results
        else:
            for model in metrics:
                for metric in metrics[model]:
                    metrics[model][metric] += fold_results[model][metric]
    for model in metrics:
        for metric in metrics[model]:
            metrics[model][metric] /= folds
    print(metrics)

if __name__ == '__main__':
    t = time()

    # Taking arg of file name, if no arg given, assumes train file is in the `data/` directory
    file_name = r"data/train.txt"
    sample_size = 1
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    if len(sys.argv) > 2:
        sample_size = float(sys.argv[2])
    
    main(file_name, sample_size=sample_size)

    print("Total time for pipeline: %0.3fsec\n" % (time()-t))

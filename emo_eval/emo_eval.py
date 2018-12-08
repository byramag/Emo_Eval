"""
Author: Abbie Byram
"""

#        -- TODO --
# - dimensionality reduction (fix)
# - adding features
#   - pinpoint emojis
#   - differentiate between turns
#   - POS, NER
#   - find emotional words
# - other classifiers
#   - Naive Bayes baseline
#   - Neural Network

import sys
import numpy as np
import pandas as pd
from time import time
from pprint import pprint
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .preprocessing import preprocess
from .feature_extraction import feature_extraction


def parseArgs(argv):
    """Takes in args from sys.argv and checks for and returns
    whatever parameters have been supplied, returning defaults
    for ones that haven't

    Authors:
        Bobby
    
    Arguments:
        argv: sys.argv
    
    Returns:
        file_name: string
            - The path to the training data
        sample_size: float (0,1]
            - Percentage of sample set to randomly sample
    """
    file_name = r"data/train.txt"
    sample_size = 1

    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    if len(sys.argv) > 2:
        sample_size = float(sys.argv[2])
    
    return file_name, sample_size


def train_crossval(x_train, y_train, folds=5):
    """TODO: documentation
    """
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_micro']
    metrics = dict()

    # # Training Naive Bayes on training data # CURRENTLY CAUSES MEMORY ERROR
    # print("Training Naive Bayes")
    # t = time()
    # nbClassifier = GaussianNB().fit(x_train_mat.toarray(), y_train)
    # print("Training Naive Bayes finished in %0.3fsec\n" % (time()-t))

    print("Training Random Forest"); t = time()
    rfClassifier = RandomForestClassifier(n_estimators=100, random_state=0)
    metrics['RF'] = cross_validate(rfClassifier, x_train, y_train, scoring=scoring, cv=folds, return_train_score=False, n_jobs=10)
    print("Training Random Forest finished in %0.3fsec\n" % (time()-t))

    print("Training Linear SVM"); t = time()
    svmLinClassifier = svm.SVC(kernel='linear', C=1)
    metrics['Linear SVM'] = cross_validate(svmLinClassifier, x_train, y_train, scoring=scoring, cv=folds, return_train_score=False, n_jobs=-1)
    print("Training Linear SVM finished in %0.3fsec\n" % (time()-t))

    #Keerthi - SGD classifier
    print("Training SGD"); t = time()
    sgd_params = {'alpha': 2.4583743122823383e-05, 'l1_ratio': 0.7561414418634068}
    sgdClassifier = SGDClassifier(max_iter=1000,penalty='elasticnet',loss='hinge',**sgd_params)
    metrics['SGD'] = cross_validate(sgdClassifier, x_train, y_train, scoring=scoring, cv=folds, return_train_score=False,n_jobs=3)
    print("Training SGD finished in %0.3fsec\n" % (time()-t))

    for model in metrics:
        for metric in metrics[model]:
            metrics[model][metric] = np.mean(metrics[model][metric])

    return metrics


def main(train_file, sample_size=1, folds=5):
    t0 = time()

    # Reading training file into dataframe
    print("Reading train file"); t = time()
    instances = pd.read_csv(train_file, sep='\t', header=0)
    print("Finished reading train file in %0.3fsec\n" % (time()-t))

    print("Sampling training data"); t = time()
    instances = instances.sample(frac=sample_size)
    print("Finished sampling training data in %0.3fsec\n" % (time()-t))

    # Apply preprocessing
    x_all, y_all = preprocess(instances)

    # Perform feature extraction
    x_all = feature_extraction(x_all)

    # Train model
    metrics = train_crossval(x_all, y_all, folds)

    print("Total time for pipeline: %0.3fsec\n" % (time()-t0))
    pprint(metrics)
    return metrics


if __name__ == '__main__':
    file_name, sample_size = parseArgs(sys.argv)
    main(file_name, sample_size=sample_size)

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
from getopt import getopt, GetoptError
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
    argv = argv[1:]

    params = {
        "file_name": r"data/train.txt",
        "sample_size": 1,
        "folds": 5,
        "tfidf": True,
        "embeddings": False,
        "emoji_vectors": False,
        "clean_data": True,
        "expand_abbrs": False,
        "desmile": False,
        "rm_stops": False,
        "adjs": False
    }

    long_args = ["input=", "cv", "no-tfidf", "embeddings", "emoji-vectors", "no-clean", "exp-abbrs", "desmile", "rm-stopwords", "only-adjs"]
    try:
        opts, _ = getopt(sys.argv[1:], "i:s", long_args)
    except GetoptError as e:
        raise ValueError(e)
    
    for opt, arg in opts:
        if opt in ('-i', '--input'):    params['file_name'] = arg
        elif opt == '-s':               params['sample_size'] = float(arg)
        elif opt == '--folds':          params['folds'] = int(arg)
        elif opt == '--no-tfidf':       params['tfidf'] = False
        elif opt == '--embeddings':     params['embeddings'] = True
        elif opt == '--emoji-vectors':  params['emoji_vectors'] = True
        elif opt == '--no-clean':       params['clean_data'] = False
        elif opt == '--exp-abbrs':      params['expand_abbrs'] = True
        elif opt == '--desmile':        params['desmile'] = True
        elif opt == '--rm-stopwords':   params['rm_stops'] = True
        elif opt == '--only-adjs':      params['adjs'] = True
    
    return params


def train_crossval(x_train, y_train, folds=5):
    """Trains Random Forest, Support Vector Machine, and Stochastic 
    Gradient Descent classifiers and evaluates using cross validation 
    with a given number of folds and returns the evaluation metrics 
    for each classifier

    Authors:
        Abbie
        Bobby
        Keerthi
    
    Arguments:
        x_train: scipy matrix of feature vectors
            - The set of features for each training instance
        y_train: string[]
            - The labels for each training instance
        folds: int
            - The number of folds for cross validation
    
    Returns:
        metrics: dictionary(classifier_name - string : scores - array of evaluation metrics)
            - The scores for each classifier
    """
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_micro']
    metrics = dict()

    # # Training Naive Bayes on training data # CURRENTLY CAUSES MEMORY ERROR (feature space too large)
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


def main(argv):
    t0 = time()

    print("Parsing command line args...", end="", flush=True); t = time()
    params = parseArgs(argv)
    print("finished %0.3fsec\n" % (time()-t))

    # Reading training file into dataframe
    print("Reading train file"); t = time()
    instances = pd.read_csv(params['file_name'], sep='\t', header=0)
    print("Finished reading train file in %0.3fsec\n" % (time()-t))

    print("Sampling training data"); t = time()
    instances = instances.sample(frac=params['sample_size'])
    print("Finished sampling training data in %0.3fsec\n" % (time()-t))

    # Apply preprocessing
    x_all, y_all = preprocess(instances,
        clean_data=params['clean_data'],
        expand_abbrs=params['expand_abbrs'],
        desmile=params['desmile'],
        remove_stopwords=params['rm_stops'],
        only_adjs=params['adjs'])

    # Perform feature extraction
    x_all = feature_extraction(x_all,
        tfidf=params['tfidf'],
        embeddings=params['embeddings'],
        emojis=params['emoji_vectors'])

    # Train model
    metrics = train_crossval(x_all, y_all,
        folds=params['folds'])

    print("Total time for pipeline: %0.3fsec\n" % (time()-t0))
    pprint(metrics)
    return metrics


if __name__ == '__main__':
    main(sys.argv)

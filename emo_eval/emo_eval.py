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

import re, sys, json
from time import time
import numpy as np
import pandas as pd
import spacy
from pprint import pprint
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
#from pandas_ml import ConfusionMatrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.tokenize import TweetTokenizer 
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

from .preprocessing import preprocess


def feature_extraction(samples):
    """TODO: documentation
    """
  # Creating bag of words feature vectors from training and test data
  # countBoW = CountVectorizer() # add stopword removal or ngrams with: ngram_range=(1,2), stop_words='english'
    
    # stop words -- lowers score
    # norm='l1' -- lowers score
    # ngram_range (1,1), (1,3), (1,4), (1,5) -- lowers score
    # min_df, max_df -- betters score, gives lower dimension
    # sublinear_tf=True -- slighty betters score
    # TweetTokenizer -- slighty betters score
    # WordNetLemmatizer -- minor or no impact
    # Dimensionality reduction -- values in the range from 5 to 5000 lowers score

    nlp = spacy.load('en_core_web_lg')

    print("\tGenerating spacy documents for samples...", end="", flush=True); t= time()
    docs = [nlp(x) for x in samples]
    print("finished in {:.3f}s".format(time()-t))

    print("\tExtracting adjectives...", end="", flush=True); t= time()
    for i, sample in enumerate(samples):
        sample = [token for token in docs[i] if token.pos_ == 'ADJ']
    print("finished in {:.3f}s".format(time()-t))

    print("\tCalculating tf-idf vectors...", end="", flush=True); t= time()
    tw = TweetTokenizer()
    wnl = WordNetLemmatizer()
    tfidf_vectors = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=10,
        max_df=0.95,
        sublinear_tf=True,
        norm='l2', 
        smooth_idf=True,
        tokenizer=lambda x:[wnl.lemmatize(t) for t in tw.tokenize(x)]
    ).fit_transform([" ".join([token.text for token in s]) for s in samples])
    print("finished in {:.3f}s".format(time()-t))

    print("\tCalculating word embedding vectors...", end="", flush=True); t= time()
    word_embeddings = [x.vector for x in samples]
    print("finished in {:.3f}s".format(time()-t))

    print("\tConcatenating matrices...", end="", flush=True); t= time()
    x_train_mat = hstack([tfidf_vectors, word_embeddings])
    print("finished in {:.3f}s".format(time()-t))

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

    return x_train_mat


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

    print("Preprocessing data"); t = time()
    x_all, y_all = preprocess(instances)
    print("Finished preprocessing in %0.3fsec\n" % (time()-t))

    print("Creating feature vectors"); t = time()
    x_all = feature_extraction(x_all)
    print("Creating feature vectors finished in %0.3fsec\n" % (time()-t))

    metrics = train_crossval(x_all, y_all, folds)

    print("Total time for pipeline: %0.3fsec\n" % (time()-t0))

    pprint(metrics)

if __name__ == '__main__':
    # Taking arg of file name, if no arg given, assumes train file is in the `data/` directory
    file_name = r"data/train.txt"
    sample_size = 1
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    if len(sys.argv) > 2:
        sample_size = float(sys.argv[2])
    
    main(file_name, sample_size=sample_size)

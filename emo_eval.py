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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
#from pandas_ml import ConfusionMatrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.tokenize import TweetTokenizer 
from nltk.stem import WordNetLemmatizer
import emoji
import emot
import nltk
nltk.download('wordnet')

# import spacy
# import en_core_web_sm
# Spacy model setup
# nlp = en_core_web_sm.load()

#Keerthi - Preprocessing start
def preprocess(instances):
    """TODO: documentation
    """
    # Separating the labels and strings into separate arrays & concatenating turns from bag of words
    row_strings = []
    labels = []
    for _, instance in instances.iterrows():
        row_strings.append(instance['turn1'] + ' ' + instance['turn2'] + ' ' + instance['turn3'])
        labels.append(instance['label'])
    
        
    CONTRACTION_MAP = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                       "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                       "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                       "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                       "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                       "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 
                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                       "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                       "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                       "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                       "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                       "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                       "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                       "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                       "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                       "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                       "this's": "this is",
                       "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                       "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                       "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                       "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                       "we're": "we are", "we've": "we have", "weren't": "were not", 
                       "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                       "what's": "what is", "what've": "what have", "when's": "when is", 
                       "when've": "when have", "where'd": "where did", "where's": "where is", 
                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                       "who's": "who is", "who've": "who have", "why's": "why is", 
                       "why've": "why have", "will've": "will have", "won't": "will not", 
                       "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                       "you'll've": "you will have", "you're": "you are", "you've": "you have" } 

    #Expands contractions in a sentence according to a contraction_mapping.
    # expand word contractions (can't -> cannot) -- minor or no impact
    
    def expand_contractions(sentence, contraction_mapping): 

        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),  
                                          flags=re.IGNORECASE|re.DOTALL) 
        def expand_match(contraction): 
            match = contraction.group(0) 
            first_char = match[0] 
            expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())   
            if not expanded_contraction:
                return match
            expanded_contraction = first_char+expanded_contraction[1:] 
            return expanded_contraction

        expanded_sentence = contractions_pattern.sub(expand_match, sentence) 
        return expanded_sentence 


    #Replaces emoticons with their meanings
    # emojis replacement -- good
    # smileys replacement -- minor or no impact
    def desmilify(text):
        
        emoticons = emot.emoticons(text)
        if type(emoticons) == dict:
            for loc,mean,value in zip(emoticons.get('location'),emoticons.get('mean'),emoticons.get('value')):
                text = text.replace(value,':%s:'%'_'.join(mean.split()))
        return text
        
    row_strings = [desmilify(emoji.demojize(expand_contractions(txt, CONTRACTION_MAP))) for txt in row_strings]

    return row_strings, labels
#Keerthi - Preprocessing end

#Keerthi - Feature reduction start
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

    tw = TweetTokenizer()
    wnl = WordNetLemmatizer()
    x_train_mat = TfidfVectorizer(ngram_range=(1,2),min_df=10,max_df=0.95, sublinear_tf=True, norm='l2', smooth_idf=True,tokenizer=lambda x:[wnl.lemmatize(t) for t in tw.tokenize(x)]).fit_transform(samples)
    
    #x_train_mat = TfidfVectorizer(ngram_range=(1,2)).fit_transform(samples)

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

#Keerthi - feature reduction end


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

    print("Training Random Forest")
    t = time()
    rfClassifier = RandomForestClassifier(n_estimators=100, random_state=0)#.fit(x_train, y_train)
    metrics['RF'] = cross_validate(rfClassifier, x_train, y_train, scoring=scoring, cv=folds, return_train_score=False, n_jobs=10)
    print("Training Random Forest finished in %0.3fsec\n" % (time()-t))

    print("Training SVM")
    t = time()
    svmClassifier = svm.SVC(kernel='linear', C=1)#.fit(x_train, y_train)
    metrics['SVM'] = cross_validate(svmClassifier, x_train, y_train, scoring=scoring, cv=folds, return_train_score=False, n_jobs=-1)
    print("Training SVM finished in %0.3fsec\n" % (time()-t))

#Keerthi - SGD classifier start
    sgd_params = {'alpha': 2.4583743122823383e-05, 'l1_ratio': 0.7561414418634068}
    sgdClassifier = SGDClassifier(max_iter=1000,penalty='elasticnet',loss='hinge',**sgd_params)

    print("Training SGD")
    t = time()
    metrics['SGD'] = cross_validate(sgdClassifier, x_train, y_train, scoring=scoring, cv=folds, return_train_score=False,n_jobs=3)
    print("Training SGD finished in %0.3fsec\n" % (time()-t))
    
#Keerthi - SGD classifier end

    for model in metrics:
        for metric in metrics[model]:
            metrics[model][metric] = np.mean(metrics[model][metric])

    return metrics


def main(train_file, sample_size=1, folds=5):
    # Reading training file into dataframe
    print("Reading train file"); t = time()
    instances = pd.read_csv(file_name, sep='\t', header=0)
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

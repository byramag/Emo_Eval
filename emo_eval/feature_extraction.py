import re
import nltk
import spacy
import numpy as np
from time import time
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def feature_extraction(samples, tfidf=True, embeddings=False, emojis=False):
    """Second stage of pipeline, takes in the string training samples and
    returns vector representations of them to be used in modeling

    Authors:
        Abbie
        Keerthi
        Bobby
    
    Arguments:
        samples: string list
            - The raw string training samples taken from preprocessing
        tfidf: boolean (True)
            - Flag to use tf-idf vectors
        embeddings: boolean (False)
            - Flag to use word embedding vectors
        emojis: boolean (False)
            - Flag to use emoji vectors
    
    Returns:
        Scipy matrix of feature vectors
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

    if not tfidf and not embeddings and not emojis:
        raise ValueError("One or more of 'tfidf', 'embeddings', and 'emojis' arguments must be True")

    print("Creating feature vectors"); t = time()

    if tfidf:
        nltk.download('wordnet')
        print("\tCalculating tf-idf vectors...", end="", flush=True); t = time()
        tw = nltk.tokenize.TweetTokenizer()
        wnl = nltk.stem.WordNetLemmatizer()
        tfidf_vectors = TfidfVectorizer(
            ngram_range=(1,2),
            min_df=10,
            max_df=0.95,
            sublinear_tf=True,
            norm='l2',
            smooth_idf=True,
            tokenizer=lambda x:[wnl.lemmatize(t) for t in tw.tokenize(x)]
        ).fit_transform(samples)
        print("finished in {:.3f}s".format(time()-t))

    if embeddings:
        nlp = spacy.load('en_core_web_lg')
        print("\tCalculating word embedding vectors...", end="", flush=True); t = time()
        word_embeddings = [nlp(x).vector for x in samples]
        print("finished in {:.3f}s".format(time()-t))
    
    if emojis:
        ev = np.empty((0,len(samples)))
        col_map = dict()
        for i, sample in enumerate(samples):
            emojis = re.findall(r'[\U00010000-\U0010ffff]', sample)
            for emoji in emojis:
                if emoji not in col_map:
                    col_map[emoji] = len(ev)
                    ev = np.append(ev, np.zeros((1,len(samples))), axis=0)
                ev[col_map[emoji]][i] += 1
        emoji_vectors = ev.T

    if tfidf and embeddings and emojis:
        print("\tConcatenating matrices...", end="", flush=True); t = time()
        x_train = hstack([tfidf_vectors, word_embeddings, emoji_vectors])
        print("finished in {:.3f}s".format(time()-t))
    elif tfidf and embeddings:
        print("\tConcatenating matrices...", end="", flush=True); t = time()
        x_train = hstack([tfidf_vectors, word_embeddings])
        print("finished in {:.3f}s".format(time()-t))
    elif tfidf and emojis:
        print("\tConcatenating matrices...", end="", flush=True); t = time()
        x_train = hstack([tfidf_vectors, emoji_vectors])
        print("finished in {:.3f}s".format(time()-t)) 
    elif embeddings and emojis:
        print("\tConcatenating matrices...", end="", flush=True); t = time()
        x_train = hstack([embeddings, emoji_vectors])
        print("finished in {:.3f}s".format(time()-t))
    elif tfidf:
        x_train = tfidf_vectors
    elif embeddings:
        x_train = word_embeddings
    elif emojis:
        x_train = emoji_vectors

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

    print("Creating feature vectors finished in %0.3fsec\n" % (time()-t))
    return x_train

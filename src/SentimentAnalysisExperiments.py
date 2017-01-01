import pandas as ps;
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from wordsegment import segment
from sklearn.linear_model import RidgeClassifier
from src.twokenize import tokenizeRawTweetText
import csv

MAX_FEATURES = 20000


def readData(fileName, encoding='utf-8'):
    with open(fileName, 'rt', encoding=encoding) as f:
        reader = csv.reader(f, delimiter='\t')
        data_as_list = list(reader)
        data_as_matrix = np.array(data_as_list)
        X = data_as_matrix[1:, 0]
        y = data_as_matrix[1:, 1]
        return X, y


def hashtagSegmenter(text):
    arr = text.split(' ')
    new_arr = []
    for elt in arr:
        if '#' in str(elt):
            new_arr.extend(segment(elt))
        else:
            new_arr.append(elt)
    return ' '.join(new_arr)


def checkAllModels(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    svm_clf = Pipeline([('vect', CountVectorizer(max_features=MAX_FEATURES,
                                                 preprocessor=hashtagSegmenter,
                                                 tokenizer=tokenizeRawTweetText)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                              alpha=1e-3, n_iter=5, random_state=42)),
                        ])
    svm_clf = svm_clf.fit(X_train, y_train)

    predicted_svm = svm_clf.predict(X_test)
    accuracy_svm = np.mean(predicted_svm == y_test)
    nb_clf = Pipeline([('vect', CountVectorizer(max_features=MAX_FEATURES,
                                                preprocessor=hashtagSegmenter,
                                                tokenizer=tokenizeRawTweetText)),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultinomialNB()),
                       ])
    nb_clf = nb_clf.fit(X_train, y_train)
    predicted_nb = nb_clf.predict(X_test)
    accuracy_nb = np.mean(predicted_nb == y_test)

    ridge_clf = Pipeline([('vect', CountVectorizer(max_features=MAX_FEATURES,
                                                   preprocessor=hashtagSegmenter,
                                                   tokenizer=tokenizeRawTweetText)),
                          ('tfidf', TfidfTransformer()),
                          ('clf', RidgeClassifier(tol=1e-2, solver="sag"))])
    ridge_clf = ridge_clf.fit(X_train, y_train)
    predicted_ridge = ridge_clf.predict(X_test)
    accuracy_ridge = np.mean(predicted_ridge == y_test)

    print("Naive Bayes Accuracy:", accuracy_nb, "\nSVM Accuracy:",
          accuracy_svm, "\nRidge Regression Accuracy:", accuracy_ridge)


def main():
    datafiles = [#'../data/kaggle-airline-tweets-data.tsv',
                 #'../data/updown-obama-data.tsv',
                 #'../data/kaggle-umich-sentences-data.tsv',
                 #'../data/sem_eval_task_b.tsv',
                 #'../data/sentiment140-data.tsv',
                 #'../data/updown-hcr-data.tsv',
                 #'../data/vader-amazon-data.tsv',
                 #'../data/vader-movies-data.tsv',
                 #'../data/vader-nyt-data.tsv',
                 '../data/vader-tweets-data.tsv',
                 '../data/sanders-data.tsv']

    for datafile in datafiles:
        if ((datafile == '../data/sanders-data.tsv') |
            (datafile == '../data/sentiment140-data.tsv')):
            X, y = readData(datafile, "ISO-8859-1")
            print(datafile)
            checkAllModels(X, y)
        else:
            X, y = readData(datafile)
            print(datafile)
            checkAllModels(X, y)

    # stemmer
    # boosted tree
    # start issuing predictions for data
    # partial fit
    # emoji
    # POS Tagger -> WordNet Lemmatizer?

    return

if __name__ == '__main__':
    main()

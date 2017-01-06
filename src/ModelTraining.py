# import required modules
#------------------------

import re, string, sqlite3
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize.casual import TweetTokenizer
from nltk.sentiment.util import mark_negation
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# define constants to be used later
#----------------------------------

DB_FILE   = "data/database.db"
ENGINE    = create_engine("sqlite:///{0}".format(DB_FILE))
RAWDATA   = 'data/vader-tweets-data.tsv'
DATE      = '2017-01-02'
THRESHOLD = 0.85

# define functions to save and load the raw training data from disk
#------------------------------------------------------------------

def loadRawData(fpath, textcol, labelcol, **kwargs):
    """load an initial training data set into memory
    :param fpath: file path to the data
    :param textcol: name of the column containing the tweet text
    :param labelcol: name of the column containing the sentiment label [-1,0,1]
    :param **kwargs: additional arguments to pass to pd.read_csv()
    :return: separate series for the text (X) and label (y)
    """

    data = pd.read_csv(fpath, usecols=[textcol, labelcol], **kwargs)
    X, y = data[textcol], data[labelcol]
    return X, y.astype(np.int32)


def saveRawData(engine, X, y):
    """save the initial training data to the database for use in later retraining
    :param engine: sqlalchemy engine for database connection
    :param X: tweet text vector
    :param y: sentiment label vector
    :return: None
    """

    data = pd.DataFrame({'tweet_text': X, 'label': y})
    data.to_sql('training', engine, if_exists='replace', index=False)
    return None


# define tweet pre-processing/utility functions
#----------------------------------------------

def tokenizeTweet(tweet):
    """tokenize a tweet according to several sequential steps
    :param tweet: tweet text as a string
    :return: tokenized tweet as a list
    1. basic steps taken by TweetTokenizer()
    2. mark negated words with [X_NEG]
    3. filter out punctuation, numbers, and one/two letter words
    4. remove common english stopwords
    """

    # initialize the baseline TweetTokenizer() and tokenize the tweet into a list of tokens
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokens    = tokenizer.tokenize(tweet)

    # mark negation for words between a negative and a punctuation character, and undo emoticon negation
    tokens = mark_negation(tokens)
    tokens = [re.sub('([' + re.escape(string.punctuation) + ']{2,})_NEG', r'\1', token) for token in tokens]

    # filter out single punctuation characters, numbers, and one/two letter words
    punct  = r'^([' + re.escape(string.punctuation) + r'])(_NEG)?$'
    numb   = r'^([\d,.]+)(_NEG)?$'
    word   = r'^\b\w{1,2}(_NEG)?\b$'
    comb   = '(' + '|'.join([punct, numb, word]) + ')'
    tokens = [token for token in tokens if not re.match(comb, token)]

    # remove common English stopwords (both negated and un-negated)
    swords = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in swords and re.sub(r'_NEG$', '', token) not in swords]

    # return the tokenized tweet as a list
    return tokens


# define model training functions
#--------------------------------

def cvTrain(X, y):
    """cross-validate and fit a SGDClassifier() on an initial training data set
    :param X: raw tweet text vector (will be transformed prior to fitting)
    :param y: sentiment label vector
    :return: tuned and fit [vectorizer] and [classifier] objects to use going forward
    """

    # set up the text vectorizer, classifier, pipeline, and tuning parameters
    vectorizer = HashingVectorizer(tokenizer=tokenizeTweet, non_negative=True, n_features=2**16)
    classifier = SGDClassifier(loss='log', penalty='l2', shuffle=True, n_iter=10, n_jobs=4)
    pipeline   = Pipeline([('vct', vectorizer), ('clf', classifier)])
    parameters = dict(vct__ngram_range=[(1,1), (1,2)], vct__norm=[None, 'l1', 'l2'],
                      vct__binary=[True, False], clf__alpha=[1e-5, 1e-4, 1e-3])

    # use cross-validation to select tuning parameters
    print("Tuning the Model on Initial Data via Cross Validation")
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, cv=5, verbose=1)
    cv.fit(X, y)

    # print the best score, best parameters, and return the fit estimator
    print("\nBest Out-of-Sample Accuracy Score: {0:4.3f}".format(cv.best_score_))
    print("Best Tuning Parameter Values:")
    for k, v in cv.best_params_.items():
        print("{0} = {1}".format(k, v))

    vectorizer = cv.best_estimator_.get_params()['vct']
    classifier = cv.best_estimator_.get_params()['clf']
    return vectorizer, classifier


def reTrain(engine, vectorizer, classifier, epochs=20):
    """fully re-train the classifier using both the original training and newly labeled data
    :param engine: sqlalchemy engine for database connection
    :param vectorizer: sklearn text vectorizer used for feature creation
    :param classifier: sklearn classifier used for classification/prediction
    :param epochs: number of training epochs (full passes through the training data)
    :return: the original [vectorizer] and updated [classifier]
    """

    o_data = pd.read_sql('SELECT tweet_text, label FROM training', engine)
    o_X, o_y = o_data.tweet_text, o_data.label

    n_data = pd.read_sql('SELECT tweet_text, label FROM labeled', engine)
    n_X, n_y = n_data.tweet_text, n_data.label

    X = vectorizer.transform(np.array(o_X.append(n_X)))
    y = np.array(o_y.append(n_y))

    classifier.set_params(verbose=1, n_iter=epochs, warm_start=True)
    return vectorizer, classifier.fit(X, y)


def batchTrain(labeled, vectorizer, classifier):
    """update the classifier using partial_fit() on a batch of newly labeled data
    :param tweets: dataframe of newly labeled tweets [tweet_text, label]
    :param vectorizer: sklearn text vectorizer used for feature creation
    :param classifier: sklearn classifier used for classification/prediction
    :return: the original [vectorizer] and updated [classifier]
    """

    X, y = vectorizer.transform(labeled.tweet_text), labeled.label
    classes = np.unique(y)

    classifier.set_params(verbose=1, warm_start=True)
    return vectorizer, classifier.partial_fit(X, y, classes)


# define functions to pull unlabeled tweets from the database and label them
#---------------------------------------------------------------------------

def pullTweets(engine, date, count=25):
    """retrieve saved tweets not yet labeled from a specific date
    :param engine: sqlalchemy engine for database connection
    :param date: string date (YYYY-MM-DD) from which to pull tweets
    :param count: number of tweets to pull per movie title
    :return: data frame of tweets or None on failure
    """

    titles = pd.read_sql("SELECT DISTINCT title FROM tweets WHERE tweet_date = '{0}'".format(date), engine).title
    tweets = []

    for title in titles:
        query = '''SELECT title, tweet_id, tweet_text
                   FROM   tweets
                   WHERE  tweet_date = "{0}"
                   AND    title = "{1}"
                   AND    tweet_id NOT IN (SELECT DISTINCT tweet_id FROM labeled)
                   LIMIT  {2}'''.format(date, title, count)
        tweets.append(pd.read_sql(query, engine))

    tweets = pd.concat(tweets, axis=0, ignore_index=True)
    if not tweets.empty:
        return tweets
    else:
        return None
    

def labelSelfTrain(engine, vectorizer, classifier, tweets, threshold):
    """label a data set of new tweets via self-training and save them to the database
    :param engine: sqlalchemy engine for database connection
    :param vectorizer: sklearn text vectorizer used for feature creation
    :param classifier: sklearn classifier used for classification/prediction
    :param tweets: a dataframe of tweets to be labeled [title, tweet_id, tweet_text]
    :param threshold: predicted probability threshold to be used for self-training
    :return: a dataframe of newly labeled tweets or None on failure
    note: tweets with max class probabilities above the threshold will be auto-labeled
    """
    
    classprobs = classifier.predict_proba(vectorizer.transform(tweets.tweet_text))
    labeled = tweets.copy()
    
    labeled['pr_neg'] = classprobs[:, 0]
    labeled['pr_neu'] = classprobs[:, 1]
    labeled['pr_pos'] = classprobs[:, 2]
    
    labeled = labeled.ix[np.amax(labeled[['pr_neg', 'pr_neu', 'pr_pos']], axis=1) >= threshold, :]
    labeled['label']  = np.argmax(np.array(labeled[['pr_neg', 'pr_neu', 'pr_pos']]), axis=1) - 1
    labeled['method'] = 'self-training'
    
    neg = labeled.ix[labeled.label == -1, :].shape[0]
    neu = labeled.ix[labeled.label ==  0, :].shape[0]
    pos = labeled.ix[labeled.label ==  1, :].shape[0]

    print("{0} Negative, {1} Neutral, and {2} Positive Tweets Labeled via Self-Training".format(neg, neu, pos))
    labeled.to_sql('labeled', engine, if_exists='append', index=False)

    if not labeled.empty:
        return labeled
    else:
        return None


def labelManual(engine, vectorizer, classifier, tweets, threshold):
    """label a data set of new tweets via manual annotation and save them to the database
    :param engine: sqlalchemy engine for database connection
    :param vectorizer: sklearn text vectorizer used for feature creation
    :param classifier: sklearn classifier used for classification/prediction
    :param tweets: a dataframe of tweets to be labeled [title, tweet_id, tweet_text]
    :param threshold: predicted probability threshold to be used for manual annotation
    :return: a dataframe of the newly labeled tweets or None on failure
    note: only tweets with max class probabilities below the threshold will be manually labeled
    """

    classprobs = classifier.predict_proba(vectorizer.transform(tweets.tweet_text))
    labeled = tweets.copy()

    labeled['pr_neg'] = classprobs[:, 0]
    labeled['pr_neu'] = classprobs[:, 1]
    labeled['pr_pos'] = classprobs[:, 2]

    labeled = labeled.ix[np.amax(labeled[['pr_neg', 'pr_neu', 'pr_pos']], axis=1) < threshold, :]
    labeled.reset_index(inplace=True, drop=True)
    labeled['label']  = np.nan
    labeled['method'] = 'manual-annotation'

    i = 0
    while True:
        try:

            print("\nTweet #{0}: {1}".format(i, labeled.tweet_text[i]))
            print("Predicted Probabilities: neg={0:4.3f} neu={1:4.3f} pos={2:4.3f}".format(labeled.pr_neg[i], labeled.pr_neu[i], labeled.pr_pos[i]))
            selection = input("Choose an Option: [-1]=Negative | [0]=Neutral | [1]=Positive | [s]=Skip | [q]=Quit\n")

            if selection.strip().lower() not in ['-1', '0', '1', 's', 'q']:
                print("Please choose a valid selection")
                continue
            elif selection.strip().lower() in ['-1', '0', '1']:
                print("Tweet manually labeled")
                labeled.ix[i, 'label'] = int(selection.strip().lower())
            elif selection.strip().lower() == 's':
                print("Tweet skipped")
                pass
            else:
                print("\nLabeling terminated by user")
                break
            i += 1

        except KeyError:
            print("\nThere are no more tweets to labeledly label")
            break

    neg = labeled.ix[labeled.label == -1, :].shape[0]
    neu = labeled.ix[labeled.label ==  0, :].shape[0]
    pos = labeled.ix[labeled.label ==  1, :].shape[0]

    print("{0} Negative, {1} Neutral, and {2} Positive Tweets Labeled via Manual Annotation".format(neg, neu, pos))
    labeled = labeled.ix[labeled.label.notnull(), :]
    labeled.to_sql('labeled', engine, if_exists='append', index=False)

    if not labeled.empty:
        return labeled
    else:
        return None


# run some unit tests with the functions defined above
#-----------------------------------------------------

# load in the raw training data and separate into [init, batch, test] data

X, y = loadRawData(RAWDATA, 'text', 'sentiment', sep='\t')
accuracies = []

X_init,    y_init    = X[:1500],     y[:1500]
X_batch_1, y_batch_1 = X[1500:2000], y[1500:2000]
X_batch_2, y_batch_2 = X[2000:2500], y[2000:2500]
X_batch_3, y_batch_3 = X[2500:3000], y[2500:3000]
X_test,    y_test    = X[3000:],     y[3000:]

saveRawData(ENGINE, X_init, y_init)
cnx = sqlite3.connect('data/database.db')
cur = cnx.cursor()
cur.execute('DELETE FROM labeled')
cnx.commit()
cnx.close()

batch_1 = pd.DataFrame({'title': 'test', 'tweet_id': np.arange(len(X_batch_1)) + 1000,   'tweet_text': X_batch_1})
batch_2 = pd.DataFrame({'title': 'test', 'tweet_id': np.arange(len(X_batch_2)) + 10000,  'tweet_text': X_batch_2})
batch_3 = pd.DataFrame({'title': 'test', 'tweet_id': np.arange(len(X_batch_3)) + 100000, 'tweet_text': X_batch_3})

# iteratively re-train and assess test set accuracy

# initial training/scoring
vectorizer, classifier = cvTrain(X_init, y_init)
accuracies.append(classifier.score(vectorizer.transform(X_test), y_test))

# re-fit after handling batch #1
labelSelfTrain(ENGINE, vectorizer, classifier, batch_1, THRESHOLD)
# labelManual(ENGINE, vectorizer, classifier, batch_1, THRESHOLD)
vectorizer, classifier = reTrain(ENGINE, vectorizer, classifier)
accuracies.append(classifier.score(vectorizer.transform(X_test), y_test))

# re-fit after handling batch #2
labelSelfTrain(ENGINE, vectorizer, classifier, batch_2, THRESHOLD)
# labelManual(ENGINE, vectorizer, classifier, batch_2, THRESHOLD)
vectorizer, classifier = reTrain(ENGINE, vectorizer, classifier)
accuracies.append(classifier.score(vectorizer.transform(X_test), y_test))

# re-fit after handling batch #3
labelSelfTrain(ENGINE, vectorizer, classifier, batch_3, THRESHOLD)
# labelManual(ENGINE, vectorizer, classifier, batch_3, THRESHOLD)
vectorizer, classifier = reTrain(ENGINE, vectorizer, classifier)
accuracies.append(classifier.score(vectorizer.transform(X_test), y_test))
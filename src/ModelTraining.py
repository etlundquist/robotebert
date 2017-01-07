# import required modules
#------------------------

import os, re, string, pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer
from nltk.sentiment.util import mark_negation
from nltk.corpus import wordnet, stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# define constants to be used later
#----------------------------------

DB_FILE   = "data/database.db"
ENGINE    = create_engine("sqlite:///{0}".format(DB_FILE))
RAWDATA   = 'data/vader-tweets-data.tsv'
FIRSTDATE = '2017-01-01'

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

def mapWordNet(tag):
    """utility function to map Penn TreeBank POS tags into WordNet POS tags
    :param tag: Penn TreeBank POS tag
    :return: WordNet POS tag
    """

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatizeTokens(tokens):
    """
    lemmatize raw tweet tokens using the inferred POS
    :param tokens: tweet text as a list of tokens
    :return: lemmatized list of tokens
    """

    lemmatizer = WordNetLemmatizer()
    tb_tokens  = pos_tag(tokens)
    wn_tokens  = [(token[0], mapWordNet(token[1])) for token in tb_tokens]
    lemmatized = [lemmatizer.lemmatize(token[0], token[1]) for token in wn_tokens]
    return lemmatized


def tokenizeTweet(tweet):
    """tokenize a tweet according to several sequential steps
    :param tweet: tweet text as a string
    :return: tokenized tweet as a list
    1. basic steps taken by TweetTokenizer()
    2. lemmatize all tokens by POS
    3. mark negated words with [X_NEG]
    4. filter out punctuation, numbers, and one/two letter words
    5. remove common english stopwords
    """

    # initialize the baseline TweetTokenizer(), tokenize, and lemmatize the tweet
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokens    = lemmatizeTokens(tokenizer.tokenize(tweet))

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
    """cross-validate and fit a TFIDFVectorizer() & LogisticRegression() on an initial training set
    :param X: raw tweet text vector (will be transformed prior to fitting)
    :param y: sentiment label vector
    :return: tuned and fit [estimator] object
    """

    # set up the text vectorizer, classifier, pipeline, and tuning parameters
    vectorizer = TfidfVectorizer(tokenizer=tokenizeTweet, norm='l2', use_idf=False)
    classifier = LogisticRegression(penalty='l2', multi_class='multinomial', solver='sag', max_iter=250)
    pipeline   = Pipeline([('vct', vectorizer), ('clf', classifier)])
    parameters = dict(vct__ngram_range=[(1,1), (1,2)], vct__max_features=[5000, 10000, 20000],
                      vct__binary=[True, False], clf__C=[0.1, 1, 10])

    # use cross-validation to select tuning parameters
    print("Tuning the Model on Initial Data via Cross Validation")
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, cv=5, verbose=1)
    cv.fit(X, y)

    # print the best score, best parameters, and return the fit estimator
    print("\nBest Out-of-Sample Accuracy Score: {0:4.3f}".format(cv.best_score_))
    print("Best Tuning Parameter Values:")
    for k, v in cv.best_params_.items():
        print("{0} = {1}".format(k, v))

    print("Returning Trained Estimator")
    return cv.best_estimator_


def reTrain(engine, estimator):
    """fully re-train the estimator using both the original training and newly labeled data
    :param engine: sqlalchemy engine for database connection
    :param estimator: sklearn pipeline estimator
    :return: the re-fit [estimator] object
    """

    o_data = pd.read_sql('SELECT tweet_text, label FROM training', engine)
    o_X, o_y = o_data.tweet_text, o_data.label

    n_data = pd.read_sql('SELECT tweet_text, label FROM labeled', engine)
    n_X, n_y = n_data.tweet_text, n_data.label

    X = np.array(o_X.append(n_X))
    y = np.array(o_y.append(n_y))

    estimator.set_params(clf__verbose=1)
    estimator = estimator.fit(X, y)
    return estimator


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
                   AND    tweet_id NOT IN (SELECT DISTINCT tweet_id FROM validation)
                   LIMIT  {2}'''.format(date, title, count)
        tweets.append(pd.read_sql(query, engine))

    tweets = pd.concat(tweets, axis=0, ignore_index=True)
    if not tweets.empty:
        return tweets
    else:
        return None
    

def labelSelfTrain(engine, estimator, tweets, threshold):
    """label a data set of new tweets via self-training and save them to the database
    :param engine: sqlalchemy engine for database connection
    :param estimator: sklearn pipeline estimator
    :param tweets: a dataframe of tweets to be labeled [title, tweet_id, tweet_text]
    :param threshold: predicted probability threshold to be used for self-training
    :return: a dataframe of newly labeled tweets or None on failure
    note: tweets with max class probabilities above the threshold will be auto-labeled
    """
    
    classprobs = estimator.predict_proba(tweets.tweet_text)
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


def labelManual(engine, estimator, tweets, threshold):
    """label a data set of new tweets via manual annotation and save them to the database
    :param engine: sqlalchemy engine for database connection
    :param estimator: sklearn pipeline estimator
    :param tweets: a dataframe of tweets to be labeled [title, tweet_id, tweet_text]
    :param threshold: predicted probability threshold to be used for manual annotation
    :return: a dataframe of the newly labeled tweets or None on failure
    note: only tweets with max class probabilities below the threshold will be manually labeled
    """

    classprobs = estimator.predict_proba(tweets.tweet_text)
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


# define functions to initialize/train the model and set up the validation data
# -----------------------------------------------------------------------------

def initModel(engine, X, y, fname='SentimentModel.p'):
    """initialize the SentimentModel using a given training set
    :param engine: sqlalchemy engine for database connection
    :param X: tweet text vector
    :param y: sentiment label vector
    :param fname: filename for the model object
    :return: the fit [estimator] object
    note: the fit estimator object is also saved to disk for re-use
    """

    estimator = cvTrain(X, y)
    saveRawData(engine, X, y)

    print("Saving Fit Estimator to Disk")
    mfile = open(os.path.join('models', fname), 'wb')
    pickle.dump(estimator, mfile)
    return estimator


def initUpdateValidation(engine, date, count=20, method='update'):
    """initialize [init] or update [update] the validation set via manual labeling of tweets
    :param engine: sqlalchemy engine for database connection
    :param date: date from which to pull tweets
    :param count: number of tweets pulled per title
    :param method: whether to initialize/replace [init] or update [update] the validation set
    :return: None
    """

    if method not in ['init', 'update']:
        raise TypeError('please choose either [init] or [update]')

    action = 'replace' if method == 'init' else 'append'
    tweets = pullTweets(engine, date, count)
    tweets['label'] = np.nan

    i = 0
    while True:
        try:

            print("\nTweet #{0}: {1}".format(i, tweets.tweet_text[i]))
            selection = input("Choose an Option: [-1]=Negative | [0]=Neutral | [1]=Positive | [s]=Skip | [q]=Quit\n")

            if selection.strip().lower() not in ['-1', '0', '1', 's', 'q']:
                print("Please choose a valid selection")
                continue
            elif selection.strip().lower() in ['-1', '0', '1']:
                print("Tweet manually labeled")
                tweets.ix[i, 'label'] = int(selection.strip().lower())
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

    neg = tweets.ix[tweets.label == -1, :].shape[0]
    neu = tweets.ix[tweets.label ==  0, :].shape[0]
    pos = tweets.ix[tweets.label ==  1, :].shape[0]

    print("{0} Negative, {1} Neutral, and {2} Positive Tweets Added to the Validation Set".format(neg, neu, pos))
    tweets = tweets.ix[tweets.label.notnull(), :]
    tweets.to_sql('validation', engine, if_exists=action, index=False)


# run some unit tests with the functions defined above
#-----------------------------------------------------

if __name__ == "__main__":

    # load the raw VADER tweet data, remove links, and save it to the database
    X, y = loadRawData(RAWDATA, 'text', 'sentiment', sep='\t')
    X = [re.sub(r'https?://[\w.-]+', '', tweet) for tweet in X]
    saveRawData(ENGINE, X, y)

    # create a small validation set to check model performance
    initUpdateValidation(ENGINE, FIRSTDATE, count=20, method='init')

    # initialize the model and score the validation set
    estimator  = initModel(ENGINE, X, y)
    validation = pd.read_sql('validation', ENGINE)
    score      = estimator.score(validation.tweet_text, validation.label)
    print("Initial Validation Accuracy: {0:4.3f} ({1} Validation Samples)".format(score, validation.shape[0]))


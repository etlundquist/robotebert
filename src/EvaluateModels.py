# import required modules
#------------------------

import numpy  as np
import pandas as pd

from vaderSentiment import vaderSentiment as vader
from sqlalchemy import create_engine

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from src.SentimentAnalysisExperiments import *

# define constants to be used later
#----------------------------------

DB_FILE      = "data/database.db"
ENGINE       = create_engine("sqlite:///{0}".format(DB_FILE))
DATAFILE     = "data/vader-tweets-data.tsv"
MAX_FEATURES = 5000

# train a logistic regression model on external data
#---------------------------------------------------

X_train, y_train = readData(DATAFILE)
classifier = Pipeline([('vect', CountVectorizer(max_features=MAX_FEATURES,
                                                preprocessor=hashtagSegmenter,
                                                tokenizer=tokenizeRawTweetText)),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(penalty='l2', C=10.0)),
                       ])

classifier = classifier.fit(X_train, y_train)

# predict classes using LogisticRegression and VADER
#---------------------------------------------------

labeled = pd.read_sql('labeled', ENGINE)
X_test  = labeled.tweet_text
y_test  = list(map(str, labeled.label))

analyzer = vader.SentimentIntensityAnalyzer()
scores   = [analyzer.polarity_scores(tweet)['compound'] for tweet in X_test]
classes  = []

for score in scores:
    if score < -0.33:
        classes.append(-1)
    elif score >= -0.33 and score <= 0.33:
        classes.append(0)
    else:
        classes.append(1)

pr_LR = classifier.predict(X_test)
pr_VD = list(map(str, classes))

metrics.accuracy_score(y_test, pr_LR)
metrics.accuracy_score(y_test, pr_VD)

classifier.fit(X_test[:600], y_test[:600])
classifier.score(X_test[600:], y_test[600:])




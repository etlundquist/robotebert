# import required modules
#------------------------

import src.GatherData as Gather
import src.ModelTraining as Training
import src.Metrics as Metrics

import pickle
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from src.ModelTraining import mapWordNet, lemmatizeTokens, tokenizeTweet

# define constants to be used later
#----------------------------------

DB_FILE   = "data/database.db"
ENGINE    = create_engine("sqlite:///{0}".format(DB_FILE))
M_FILE    = 'models/SentimentModel.p'
COUNT     = 20
THRESHOLD = 0.90

# define functions to update the backend
#---------------------------------------

def updateModel(engine, estimator, date, count, threshold=0.90):
    """label additional downloaded data and re-fit the model
    :param engine: sqlalchemy engine for database connection
    :param estimator: sklearn pipeline estimator
    :param date: string date (YYYY-MM-DD) from which to pull tweets
    :param count: number of tweets to pull per movie title
    :param threshold: self-training prediction threshold
    :return: the re-fit estimator or None on failure
    """

    tweets = Training.pullTweets(engine, date, count)
    if tweets.empty:
        print("No New Additional Tweets to Label")
        return None

    Training.labelSelfTrain(engine, estimator, tweets, threshold)
    Training.labelManual(engine, estimator, tweets, threshold)

    validation = pd.read_sql('validation', engine)
    old_score  = estimator.score(validation.tweet_text, validation.label)
    estimator  = Training.reTrain(engine, estimator)
    new_score  = estimator.score(validation.tweet_text, validation.label)

    print("Old Validation Accuracy: {0:4.4f} | New Validation Accuracy: {0:4.4f}".format(old_score, new_score))
    return estimator


def main():
    """gather the next day's worth of data, calculate performance metrics, and update the model"""

    try:
        date = Gather.gatherData()
    except (Gather.BOError, Gather.TweetError) as err:
        print(err.message)
        print("No New Data to Gather - Nothing Updated")
        return None

    try:
        mhandle = open(M_FILE, 'rb')
        estimator = pickle.load(mhandle)
    except OSError:
        print("Couldn't Load the Model from Disk - Metrics/Model not Updated")
        return None
    finally:
        if mhandle: mhandle.close()

    Metrics.calcMetrics(ENGINE, estimator, date)
    estimator = updateModel(ENGINE, estimator, date, COUNT, THRESHOLD)

    try:
        mhandle = open(M_FILE, 'wb')
        pickle.dump(estimator, mhandle)
    except OSError:
        print("Couldn't Write the Model to Disk - Model not Updated")
        return None
    finally:
        if mhandle: mhandle.close()

    print("\nBackend Successfully updated for {0}".format(date))
    return None

# run the main() function and update everything
#----------------------------------------------

if __name__ == "__main__":
    main()


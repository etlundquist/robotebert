# import required modules
#------------------------

import numpy  as np
import pandas as pd
from sqlalchemy import create_engine

# define constants to be used later
#----------------------------------

DB_FILE = "data/database.db"
ENGINE  = create_engine("sqlite:///{0}".format(DB_FILE))

# define functions to attach labels based on predicted probabilities or human annotations
#----------------------------------------------------------------------------------------

def pullTweets(engine, date, limit=None):
    """retrieve all saved tweets not yet labeled from a specific date
    :param engine: sqlalchemy engine for database connection
    :param date: string date (YYYY-MM-DD) for which to pull tweets
    :param limit: limit the number of tweets returned
    :return: data frame of tweets or None on failure
    """

    query = '''SELECT title, tweet_id, tweet_text
               FROM   tweets
               WHERE  tweet_date = '{0}'
               AND    tweet_id NOT IN (SELECT DISTINCT tweet_id FROM labeled)'''.format(date)

    if limit: query += " LIMIT {0}".format(limit)
    tweets = pd.read_sql(query, engine)

    if tweets.shape[0] == 0:
        print("There are no saved tweets from {0} yet to be labeled".format(date))
        return None
    else:
        return tweets


def labelProba(tweets, classprobs, threshold):
    """add labels to tweets with class probabilities over a given threshold
    :param tweets: a dataframe of tweets to be labeled [title, tweet_id, tweet_text]
    :param classprobs: an [Nx3] array of predicted class probabilities [neg, neu, pos]
    :param threshold: tweets with a max class probability of at least this value will be labeled
    :return: None
    """

    if tweets.shape[0] == 0 or classprobs.shape[0] == 0:
        print("There are no observations to label")
        return None
    if classprobs.shape[1] != 3:
        print("The classprobs array has an incorrect number of columns")
        return None
    if tweets.shape[0] != classprobs.shape[0]:
        print("The tweets and classprobs arrays don't have the same number of rows")
        return None

    indices = np.where(np.amax(classprobs, axis=1) >= threshold)
    labels  = np.argmax(classprobs[indices], axis=1) - 1

    if indices[0].shape[0] == 0:
        print("There are no tweets with a max class probability of at least {0}".format(threshold))
        return None

    labeled = tweets.ix[indices]
    labeled['pr_neg'] = classprobs[indices][:, 0]
    labeled['pr_neu'] = classprobs[indices][:, 1]
    labeled['pr_pos'] = classprobs[indices][:, 2]
    labeled['label']  = labels
    labeled['method'] = 'proba'

    cnt_neg = labeled.ix[labeled.label == -1, :].shape[0]
    cnt_neu = labeled.ix[labeled.label ==  0, :].shape[0]
    cnt_pos = labeled.ix[labeled.label ==  1, :].shape[0]

    try:
        labeled.to_sql('labeled', ENGINE, if_exists='append', index=False)
        print("{0} tweets have a predicted class probability of at least {1}".format(labeled.shape[0], threshold))
        print("Adding {0} negative, {1} neutral, and {2} positive labeled tweets".format(cnt_neg, cnt_neu, cnt_pos))
    except Exception:
        print("There was a problem writing labeled tweets to the database")


def labelManual(tweets, classprobs, minprob=0.0, maxprob=1.0):
    """label tweets with manual review using predicted probabilities and raw tweet text
    :param tweets: a dataframe of tweets to be labeled [title, tweet_id, tweet_text]
    :param classprobs: an [Nx3] array of predicted class probabilities [neg, neu, pos]
    :param minprob: only show tweets with at least this minimum probability for the most likely class
    :param maxprob: only show tweets with at most this maximum probability for the most likely class
    :return: None
    """

    if tweets.shape[0] == 0 or classprobs.shape[0] == 0:
        print("There are no observations to label")
        return None
    if classprobs.shape[1] != 3:
        print("The classprobs array has an incorrect number of columns")
        return None
    if tweets.shape[0] != classprobs.shape[0]:
        print("The tweets and classprobs arrays don't have the same number of rows")
        return None

    labeled = tweets
    labeled['pr_neg'] = classprobs[:, 0]
    labeled['pr_neu'] = classprobs[:, 1]
    labeled['pr_pos'] = classprobs[:, 2]
    labeled['label']  = np.nan
    labeled['method'] = 'manual'

    i = 0
    while True:
        try:

            highprob = np.max(classprobs[i, :])
            if (highprob < minprob) or (highprob > maxprob):
                i += 1
                continue

            print("\nTweet #{0}: {1}".format(i, labeled.tweet_text[i]))
            print("Predicted Probabilities: neg={0:4.3f} neu={1:4.3f} pos={2:4.3f}".format(classprobs[i, 0], classprobs[i, 1], classprobs[i, 2]))
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
                print("\nSession terminated by user")
                break
            i += 1

        except KeyError:
            print("\nThere are no more tweets to label")
            break

    labeled = labeled.ix[labeled.label.notnull(), :]
    cnt_neg = labeled.ix[labeled.label == -1, :].shape[0]
    cnt_neu = labeled.ix[labeled.label ==  0, :].shape[0]
    cnt_pos = labeled.ix[labeled.label ==  1, :].shape[0]

    try:
        labeled.to_sql('labeled', ENGINE, if_exists='append', index=False)
        print("{0} tweets manually labeled".format(labeled.shape[0]))
        print("Adding {0} negative, {1} neutral, and {2} positive labeled tweets".format(cnt_neg, cnt_neu, cnt_pos))
    except Exception:
        print("There was a problem writing labeled tweets to the database")

# do some unit tests with the functions defined above [NOTE: requires a way to generate classprobs]
#--------------------------------------------------------------------------------------------------

# date       = '2017-01-02'
# tweets     = pullTweets(ENGINE, date)
# ttext      = tweets.tweet_text
# classprobs = classifier.predict_proba(ttext)
# threshold  = 0.90
# labelProba(tweets, classprobs, threshold)
# labelManual(tweets, classprobs)

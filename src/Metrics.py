# import required modules
#------------------------

import sqlite3
import numpy as np
import pandas as pd
import src.ModelTraining as Training

# define functions to update the model and calculate metrics
#-----------------------------------------------------------

def mainMetrics(df):
    """calculate performance metrics via DF.groupby()"""

    pr_cls_neg = df.ix[df.pr_class == -1, :].shape[0] / df.shape[0]
    pr_cls_neu = df.ix[df.pr_class ==  0, :].shape[0] / df.shape[0]
    pr_cls_pos = df.ix[df.pr_class ==  1, :].shape[0] / df.shape[0]

    avg_pr_neg = df.pr_prob_neg.mean()
    avg_pr_neu = df.pr_prob_neu.mean()
    avg_pr_pos = df.pr_prob_pos.mean()

    results = pd.DataFrame({'title': np.unique(df.title),
                            'pr_cls_neg': pr_cls_neg, 'pr_cls_neu': pr_cls_neu, 'pr_cls_pos': pr_cls_pos,
                            'avg_pr_neg': avg_pr_neg, 'avg_pr_neu': avg_pr_neu, 'avg_pr_pos': avg_pr_pos})
    return results


def calcMetrics(engine, estimator, date):
    """calculate performance metrics for a given date using the current estimator
    :param engine: sqlalchemy engine for database connection
    :param estimator: sklearn pipeline estimator
    :param date: string date (YYYY-MM-DD) from which to pull tweets
    :return: None
    """

    tweets = Training.pullTweets(engine, date, 1e6)
    if tweets.empty:
        print("No New Additional Tweets for which to calculate metrics")
        return None

    pr_class = estimator.predict(tweets.tweet_text)
    pr_prob  = estimator.predict_proba(tweets.tweet_text)

    tweets['pr_class']    = pr_class
    tweets['pr_prob_neg'] = pr_prob[:, 0]
    tweets['pr_prob_neu'] = pr_prob[:, 1]
    tweets['pr_prob_pos'] = pr_prob[:, 2]

    results = tweets.groupby('title').apply(mainMetrics)
    results['pred_date'] = date

    try:
        results.to_sql('metrics', engine, if_exists='append', index=False)
        print("Metrics for {0} Added to the Database".format(date))
    except sqlite3.IntegrityError:
        print("Metrics for {0} Already Calculated".format(date))
    return None

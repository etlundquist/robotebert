# import required modules and set options
#----------------------------------------

import re
import json
import tweepy
import pandas as pd
from datetime import datetime

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 180)

# define functions to search twitter for movie titles and process tweets
#-----------------------------------------------------------------------

def generateAPI(c_key, c_sec, a_tok, a_sec, **kwargs):
    """generate a tweepy API object to perform actions
    :param c_key: application consumer key
    :param c_sec: application consumer secret
    :param a_tok: user access token
    :param a_sec: user secret access token
    :param kwargs: additional arguments passed to the tweepy.API() call
    :return: tweepy API object
    list of valid kwargs provided here: http://docs.tweepy.org/en/v3.5.0/api.html
    """

    auth = tweepy.OAuthHandler(c_key, c_sec)
    auth.set_access_token(a_tok, a_sec)
    api = tweepy.API(auth, **kwargs)
    return api


def processTitle(title):
    """clean up raw titles taken from BOMojo to facilitate better searches
    :param title: raw title taken from BOMojo
    :return: cleaned up string title
    removes parenthetical years and after-colon text from the titles
    subjectively it seems like most people won't include these in twitter mentions
    """

    cleaned = title.lower()
    cleaned = re.sub(r'\(\d{4}.*\)', '', cleaned)
    cleaned = re.sub(r':.+', '', cleaned).strip()
    return cleaned


def processTweet(title, tweet, remove_title=False):
    """process a given tweet to prepare it for storage
    :param title: title of the movie being searched
    :param tweet: tweepy Status object to process
    :param remove_title: whether or not to remove the title from the tweet text
    :return: dictionary of author/tweet information
    """

    # create a title regex and initialize a dictionary to hold results

    texp    = r"#?" + r" ?".join(title.split(" "))
    results = {}

    # retrieve author metadata

    results['author_id']        = tweet.author.id
    results['author_name']      = tweet.author.name
    results['author_verified']  = tweet.author.verified
    results['author_followers'] = tweet.author.followers_count
    results['author_friends']   = tweet.author.friends_count
    results['author_favorites'] = tweet.author.favourites_count
    results['author_statuses']  = tweet.author.statuses_count

    # retrieve tweet metadata

    results['tweet_id']        = tweet.id
    results['tweet_datetime']  = tweet.created_at.strftime('%Y-%m-%d %H:%m:%S')
    results['tweet_favorites'] = tweet.favorite_count
    results['tweet_retweets']  = tweet.retweet_count

    retweet = re.search('^RT @\w+:', tweet.text)
    results['tweet_retweet'] = True if retweet else False

    mention = re.search('@\w+', tweet.text)
    results['tweet_mention'] = True if mention and not retweet else False

    # retrieve raw tweet text and clean it up

    text = tweet.text.replace('\n', '').replace("'", "").replace('"', '').lower()
    text = re.sub(r'(rt )?@\w+:?', '', text)
    text = re.sub(texp, '', text) if remove_title else text
    text = re.sub(r'\&\w+;', '', text)
    text = re.sub(r' {2,}', ' ', text).strip()

    results['tweet_text'] = text
    return results


def searchMovie(api, title, date, count, retweets=False):
    """search twitter for tweets mentioning a given movie title
    :param api: tweepy API object to use for searching
    :param title: title of the movie being searched
    :param date: date used to filter tweets (YYYY-MM-DD)
    :param count: max number of tweets to return
    :param retweets: whether or not to include retweets when searching
    :return: dataframe of processed tweets
    for the time being tweets with links are excluded from the search - these are often from
    organizational accounts and/or for marketing purposes and don't reflect individual sentiment
    """

    query = "\"{0}\" since:{1} -filter:links".format(title, date)
    if retweets == False:
        query += " -filter:retweets"

    rawtweets = tweepy.Cursor(api.search, q=query, result_type="recent", lang="en").items(count)
    results   = []

    for i, tweet in enumerate(rawtweets):
        try:
            results.append(processTweet(title, tweet))
        except tweepy.error.TweepError as err:
            print("\nThere was an error processing tweet #{0}".format(i))
            print(err.messages[0]['code'])

    results = pd.DataFrame(results)
    results['movie_title'] = title
    results['search_date'] = date
    return results

# run some module tests with the functions defined above
#-------------------------------------------------------

# search for a given title on a given date and return results in a dataframe

credentials = json.loads(open('auth/twitter.json', 'r').read())
api = generateAPI(wait_on_rate_limit=True, wait_on_rate_limit_notify=True, **credentials)
res = searchMovie(api, processTitle('nocturnal animals'), '2016-12-22', 100)

res.shape
res.dtypes
res.head()
res.tweet_text.head(100)

# check resource usage status and rate limit period

api.rate_limit_status()['resources']['search']['/search/tweets']
datetime.fromtimestamp(api.rate_limit_status()['resources']['search']['/search/tweets']['reset'])
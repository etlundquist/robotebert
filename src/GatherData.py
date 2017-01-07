# import required modules
#------------------------

import re, csv, json, sqlite3, requests, tweepy
import pandas as pd
import urllib.request as urlreq
import urllib.error as urlerr

from sqlalchemy import create_engine
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from src.CustomExceptions import *

# define constants to be used later
#----------------------------------

CREDENTIALS   = json.loads(open("auth/twitter.json", "r").read())
BO_ENDPOINT   = "http://www.boxofficemojo.com/daily/chart/?view=1day&sortdate={0}&p=.htm"
OMDB_ENDPOINT = "http://www.omdbapi.com/"
DB_FILE       = "data/database.db"
ENGINE        = create_engine("sqlite:///{0}".format(DB_FILE))
FIRSTDATE     = '2017-01-01' # first date of data collection
MAX_TWEETS    = 200          # max number of tweets to gather for each movie title
CNT_MOVIES    = 10           # number of movies (by descending revenue) to take from daily box office

# define some general utility functions
#--------------------------------------

def processTitle(title):
    """clean up raw titles taken from BOMojo to facilitate better searches on OMDB/Twitter
    :param title: raw title taken from BOMojo
    :return: cleaned up string title
    note: removes parenthetical year and after-colon text from the titles
    """

    cleaned = re.sub(r'[@#\"]+', '', title.lower().strip())
    cleaned = re.sub(r'\(\d{4}.*\)', '', cleaned)
    cleaned = re.sub(r':.+', '', cleaned).strip()
    return cleaned


def outputData(tname):
    """output a data table to CSV for visual inspection
    :param tname: string name of the table to output
    :return: None
    """

    table = pd.read_sql("SELECT * FROM {0}".format(tname), ENGINE)
    table.to_csv("data/{0}.csv".format(tname), sep=",", header=True, index=False, quoting=csv.QUOTE_NONNUMERIC)


# define functions to scrape data from BoxOfficeMojo
#---------------------------------------------------

def findNextDate(cnx, firstdate):
    """pull the next date for which to search for new data
    :param cnx: database connection
    :param firstdate: first date for which to search for data [YYYY-MM-DD]
    :return: date for which to search for new data [YYYY-MM-DD]
    """

    cur = cnx.cursor()
    cur.execute("SELECT gross_date FROM boxoffice ORDER BY gross_date DESC LIMIT 1")

    try:
        lastdate = cur.fetchone()[0]
        nextdate = datetime.strptime(lastdate, '%Y-%m-%d') + timedelta(days=1)
        nextdate = nextdate.strftime('%Y-%m-%d')
    except TypeError:
        nextdate = firstdate
    finally:
        cur.close()
    return nextdate


def getTopMovies(endpoint, date, count=10):
    """scrape BOMojo for information on the top N movies in terms of daily gross
    :param endpoint: the url of the BOMojo daily gross table
    :param date: date for which to pull information
    :param count: number of movies (descending by revenue) to pull
    :return: dataframe of movie information
    note: the endpoint should have a str.format() placeholder for the date, e.g.
    note: current endpoint: http://www.boxofficemojo.com/daily/chart/?view=1day&sortdate={0}&p=.htm
    """

    try:
        response = urlreq.urlopen(endpoint.format(date))
        soup     = BeautifulSoup(response.read(), "html.parser")
        table    = soup.find('table', border="0", cellpadding="5", cellspacing="1")
        tdata    = []

        for i, row in enumerate(table.find_all('tr')[1:], start=1):
            if i > count:
                break

            cells = row.find_all('td')
            tdict = {}

            tdict['rank']         = i
            tdict['title']        = cells[2].text.strip()
            tdict['daily_gross']  = int(re.sub(r'[^\d]', '', cells[4].text))
            tdict['theaters']     = int(re.sub(r'[^\d]', '', cells[7].text))
            tdict['todate_gross'] = int(re.sub(r'[^\d]', '', cells[9].text))
            tdict['release_day']  = int(cells[10].text)

            tdata.append(tdict)

        tdata = pd.DataFrame(tdata)
        tdata['gross_date'] = date
        return tdata

    except urlerr.URLError as err:
        print("\nThere was an error retrieving daily revenue information")
        print(err)
        return None
    except Exception:
        print("\nThere's something wrong with the BOMojo daily revenue page")
        return None

# define functions to pull data from the Twitter API
#---------------------------------------------------

def generateAPI(c_key, c_sec, a_tok, a_sec, **kwargs):
    """generate a tweepy API object to perform actions
    :param c_key: application consumer key
    :param c_sec: application consumer secret
    :param a_tok: user access token
    :param a_sec: user secret access token
    :param kwargs: additional arguments passed to the tweepy.API() call
    :return: tweepy API object
    note: list of valid kwargs provided here: http://docs.tweepy.org/en/v3.5.0/api.html
    """

    auth = tweepy.OAuthHandler(c_key, c_sec)
    auth.set_access_token(a_tok, a_sec)
    api = tweepy.API(auth, **kwargs)
    return api


def processTweet(title, tweet, remove_title=False):
    """process a given tweet to prepare it for storage
    :param title: title of the movie being searched
    :param tweet: tweepy Status object to process
    :param remove_title: whether or not to remove the title from the tweet text
    :return: dictionary of author/tweet information
    """

    # create a title regex and initialize a dictionary to hold results

    texp    = r"#?" + r" ?".join(processTitle(title).split(" "))
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

    text = tweet.text.replace('\n', '').replace("'", "").replace('"', '')
    text = re.sub(r'(RT )?@\w+:?', '', text)
    text = re.sub(texp, '', text, flags=re.IGNORECASE) if remove_title else text
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
    note: for the time being tweets with links are excluded from the search - these are often from
    note: organizational accounts and/or for marketing purposes and don't reflect individual sentiment
    """

    since = date
    until = datetime.strptime(since, '%Y-%m-%d') + timedelta(days=1)
    until = until.strftime('%Y-%m-%d')

    query = "\"{0}\" since:{1} until:{2} -filter:links".format(processTitle(title), since, until)
    if retweets == False:
        query += " -filter:retweets"

    rawtweets = tweepy.Cursor(api.search, q=query, result_type="recent", lang="en").items(count)
    results   = []

    for i, tweet in enumerate(rawtweets):
        try:
            results.append(processTweet(title, tweet, remove_title=True))
        except tweepy.error.TweepError as err:
            print("\nThere was an error processing tweet #{0} for title [{1}]".format(i, title))
            print(err.messages[0]['code'])

    results = pd.DataFrame(results)
    results['title']      = title
    results['tweet_date'] = date
    return results


# define functions to pull data from the OMDB API
#------------------------------------------------

def getMovieInfo(endpoint, title, year):
    """retrieve movie metadata from OMDB
    :param endpoint: API endpoint to sent requests
    :param title: movie title to search
    :param year: movie release year
    :return: a dictionary of metadata results
    """

    params   = {'t': title, 'y': year, 'plot':'short', 'r':'json', 'tomatoes':'true'}
    response = requests.get(endpoint, params=params)

    try:
        response.raise_for_status()
        response = response.json()

        if 'Error' in response.keys():
            raise LookupError

        results  = {}
        strkeys  = ['Actors', 'Director', 'Genre', 'Plot', 'Rated', 'Released', 'imdbID', 'tomatoConsensus']
        intkeys  = ['Runtime', 'Metascore', 'imdbVotes', 'tomatoMeter', 'tomatoReviews']
        fltkeys  = ['imdbRating']

        for key in strkeys:
            results[key] = response[key] if response[key] != 'N/A' else None
        for key in intkeys:
            results[key] = int(re.sub(r'[^\d]', '', response[key])) if response[key] != 'N/A' else None
        for key in fltkeys:
            results[key] = float(re.sub(r'[^\d]', '', response[key])) if response[key] != 'N/A' else None
        return results

    except requests.exceptions.HTTPError:
        print("There was a problem with the HTTP request: {0}".format(response.status_code))
    except requests.exceptions.Timeout:
        print("The HTTP request timed out")
    except LookupError:
        pass
    return None


def insertMovie(cnx, title, date, results):
    """insert a new movie into the movies table
    :param cnx: database connection
    :param title: movie title
    :param date: search date
    :param results: getMovieInfo() results
    :return: None
    """

    try:
        cur = cnx.cursor()
        cur.execute('INSERT INTO movies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (title, date, results['Actors'], results['Director'], results['Genre'], results['Plot'],
                     results['Rated'], results['Released'], results['Runtime'], results['Metascore'], results['imdbID'],
                     results['imdbRating'], results['imdbVotes'], results['tomatoConsensus'], results['tomatoMeter'],
                     results['tomatoReviews']))
        print("Movie Info for [{0}] Written to Database".format(title))
    except sqlite3.Error:
        print('There was an error inserting movie into the database')
    finally:
        cur.close()


# define a driver function to gather new daily data
#--------------------------------------------------

def gatherData():
    """main driver program to gather daily data
    :return: date for which data has been gathered
    1. daily box office data from BoxOfficeMojo
    2. daily tweets for each title taken from the daily box office charts
    3. detailed movie metadata from OMDB for any new movie appearing on the daily charts
    note: should only be run once a day or primary key constraints will fail
    """

    # connect to database, set up the tweepy API object, and find the next date to search

    cnx = sqlite3.connect(DB_FILE)
    api = generateAPI(wait_on_rate_limit=True, wait_on_rate_limit_notify=True, **CREDENTIALS)

    nextdate = findNextDate(cnx, FIRSTDATE)
    year = nextdate[:4]

    # attempt to scrape box office data

    bodata = getTopMovies(BO_ENDPOINT, nextdate, CNT_MOVIES)

    if not bodata.empty:
        bodata.to_sql('boxoffice', ENGINE, if_exists='append', index=False)
        print("Box Office Data for [{0}] Written to Database".format(nextdate))
    else:
        raise BOError("Error Scraping/Writing Box Office Data for [{0}]".format(nextdate))

    # attempt to collect tweet data

    for movie in bodata.title:
        try:
            tweets = searchMovie(api, movie, nextdate, MAX_TWEETS)
            if not tweets.empty:
                tweets.to_sql('tweets', ENGINE, if_exists='append', index=False)
                print("Tweets for [{0}] Written to Database".format(movie))
            else:
                raise TweetError("Error Fetching/Writing Tweets for [{0}]".format(movie))
        except tweepy.error.TweepError:
            raise TweetError("Error Fetching/Writing Tweets for [{0}]".format(movie))

    # attempt to collect movie metadata

    for movie in bodata.title:
        minfo = getMovieInfo(OMDB_ENDPOINT, processTitle(movie), year)
        if minfo:
            insertMovie(cnx, movie, nextdate, minfo)
        else:
            minfo = getMovieInfo(OMDB_ENDPOINT, processTitle(movie), str(int(year)-1))
            if minfo:
                insertMovie(cnx, movie, nextdate, minfo)
            else:
                print("Movie: [{0}] Could Not be Found via OMDB".format(movie))

    # commit changes and close DB connection

    cnx.commit()
    cnx.close()

    print("\nAll Data for {0} Successfully Added to the Database!\n".format(nextdate))
    return nextdate


# execute the main data gathering function and output updated tables to CSV
#--------------------------------------------------------------------------

if __name__ == "__main__":
    gatherData()


import os
import sqlite3

# create the database file and a connection cursor
#-------------------------------------------------

if os.path.exists('data/database.db') and os.path.isfile('data/database.db'):
    os.remove('data/database.db')

cnx = sqlite3.connect('data/database.db')
cur = cnx.cursor()

# create the needed database tables
#----------------------------------

boxoffice = '''CREATE TABLE boxoffice (
                 gross_date   TEXT,
                 rank         INTEGER,
                 title        TEXT,
                 daily_gross  INTEGER,
                 todate_gross INTEGER,
                 release_day  INTEGER,
                 theaters     INTEGER,
                 PRIMARY KEY (gross_date, rank)
               )'''

movies = '''CREATE TABLE movies (
              title           TEXT,
              search_date     TEXT,
              actors          TEXT,
              director        TEXT,
              genres          TEXT,
              plot            TEXT,
              mpaarating      TEXT,
              release         TEXT,
              runtime         INTEGER,
              metascore       INTEGER,
              imdbid          TEXT,
              imdbrating      REAL,
              imdbvotes       INTEGER,
              tomatoconsensus TEXT,
              tomatometer     INTEGER,
              tomatoreviews   INTEGER,
              PRIMARY KEY (title, search_date)
            )'''

tweets = '''CREATE TABLE tweets (
              title            TEXT,
              author_id        INTEGER,
              author_name      TEXT,
              author_friends   INTEGER,
              author_followers INTEGER,
              author_favorites INTEGER,
              author_statuses  INTEGER,
              author_verified  INTEGER,
              tweet_id         INTEGER,
              tweet_date       TEXT,
              tweet_datetime   TEXT,
              tweet_favorites  INTEGER,
              tweet_retweets   INTEGER,
              tweet_retweet    INTEGER,
              tweet_mention    INTEGER,
              tweet_text       TEXT,
              PRIMARY KEY (title, tweet_id)
            )'''

labeled = '''CREATE TABLE labeled (
               title      TEXT,
               tweet_id   INTEGER,
               tweet_text TEXT,
               pr_neg     REAL,
               pr_neu     REAL,
               pr_pos     REAL,
               label      INTEGER,
               method     TEXT,
               PRIMARY KEY (title, tweet_id)
            )'''

cur.execute(boxoffice)
cur.execute(movies)
cur.execute(tweets)
cur.execute(labeled)

# commit changes and close the connection
#----------------------------------------

cur.close()
cnx.commit()
cnx.close()

# import required modules and set options
#----------------------------------------

import re
import pandas as pd
import urllib.request as urlreq
import urllib.error as urlerr
from bs4 import BeautifulSoup

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 180)

# define functions to scrape box office data
#-------------------------------------------

def findLastDate(endpoint):
    """find the date of the most recent BoxOfficeMojo daily numbers
    :param endpoint: URL of the BOMojo daily listings
    :return: string date in YYYY-MM-DD
    """

    try:
        response = urlreq.urlopen(endpoint)
        soup     = BeautifulSoup(response.read(), "html.parser")
        links    = soup.find_all('a', href=re.compile(r"/daily/chart/\?sortdate=\d{4}-\d{2}-\d{2}&p=\.htm"))
        lastdate = re.match(r".+(\d{4}-\d{2}-\d{2}).+", links[-1]['href']).group(1)
        return lastdate
    except urlerr.URLError as err:
        print("\nThere was an error retrieving the daily BOMojo chart")
        print(err)
        return None


def getTopMovies(endpoint, date, count=10):
    """scrape BOMojo for information on the topN movies in terms of daily gross
    :param endpoint: the url of the BOMojo daily gross table
    :param date: date for which to pull information
    :param count: number of movies (descending by revenue) to pull
    :return: dataframe of movie information
    the endpoint should have a str.format() placeholder for the date, e.g.
    http://www.boxofficemojo.com/daily/chart/?view=1day&sortdate={0}&p=.htm
    """

    try:
        response = urlreq.urlopen(endpoint.format(date))
        soup     = BeautifulSoup(response.read(), "html.parser")
        table    = soup.find_all('table')[8]
        tdata    = []

        for i, row in enumerate(table.find_all('tr')[1:], start=1):

            if i > count:
                break

            cells = row.find_all('td')
            tdict = {}

            tdict['rank']         = i
            tdict['title']        = cells[2].text.strip()
            tdict['studio']       = cells[3].text.strip()
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

# run some module tests with the functions defined above
#-------------------------------------------------------

endpoint_1 = "http://www.boxofficemojo.com/daily/"
date = findLastDate(endpoint_1)
endpoint_2 = "http://www.boxofficemojo.com/daily/chart/?view=1day&sortdate={0}&p=.htm".format(date)
bodata = getTopMovies(endpoint_2, date, 40)

bodata.shape
bodata.dtypes
bodata
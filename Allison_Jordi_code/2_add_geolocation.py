# -*- coding: utf-8 -*-
import sys
from urllib import urlopen
from bs4 import BeautifulSoup
import re
from geopy.geocoders import Nominatim
geolocator = Nominatim()
from geopy.exc import GeocoderTimedOut
URL_INIT = 'https://twitter.com/'
import pandas as pd
import os
import requests
from lxml import html

def parse_url(tweet_user):
    url = "{}{}".format(URL_INIT, tweet_user)
    return url


#folder_names = ['/Users/nurbektazhimbetov/Desktop/fall 2018/project/CS221-Course-project/twint_/blizzard', \
#                '/Users/nurbektazhimbetov/Desktop/fall 2018/project/CS221-Course-project/twint_/california_wildfire', \
#                '/Users/nurbektazhimbetov/Desktop/fall 2018/project/CS221-Course-project/twint_/hurricane_florence'], \
#                '/Users/nurbektazhimbetov/Desktop/fall 2018/project/CS221-Course-project/twint_/hurricane_michael']
folder_names = ['/Users/nurbektazhimbetov/Desktop/fall 2018/project/CS221-Course-project/twint_/california_wildfire']

for folder in folder_names:
    tweet_file = os.path.join(folder, 'tweets.csv')
    df = pd.read_csv(tweet_file)
    list_of_users = df['username']
    LONG_17 = ['']*len(list_of_users)
    LATT_17 = ['']*len(list_of_users)
    CITY = ['']*len(list_of_users)
    df['LONG'] = LONG_17
    df['LATT'] = LATT_17
    long_back_up_list = ['0']*len(list_of_users)
    latt_back_up_list = ['0']*len(list_of_users)

    for i in range(len(list_of_users)):
        print i
        try:
            url = parse_url(list_of_users[i])
            req = requests.get(url)
        except:
            continue
        soup = BeautifulSoup(req.content, 'html.parser')
        if not soup.find('span', 'ProfileHeaderCard-locationText'):
            continue
        location = soup.find('span', 'ProfileHeaderCard-locationText').text.encode('utf8').strip('\n').strip()
        if location:
            if ',' in location:
                splitted_location = location.split(',')
            else:
                splitted_location = re.split('|;|-|/|°|#', location)
            try:
                if splitted_location:
                    try:
                        location_location = geolocator.geocode(splitted_location[0], timeout=100)
                        if location_location:
                            df.loc[[i], 'LONG'] = location_location[1][1]
                            df.loc[[i], 'LATT'] = location_location[1][0]
                            long_back_up_list[i] = location_location[1][1]
                            latt_back_up_list[i] = location_location[1][0]
                    except:
                        pass
                else:
                    location_location = geolocator.geocode(location, timeout=100)
                    if location_location:
                        df.loc[[i], 'LONG'] = location_location[1][1]
                        df.loc[[i], 'LATT'] = location_location[1][0]
                        long_back_up_list[i] = location_location[1][1]
                        latt_back_up_list[i] = location_location[1][0]
            except GeocoderTimedOut as e:
                print("Error: geocode failed on input %s with message %s"%(location, e))
        
        with open('long_back_up.txt', 'a') as f:
            f.write("%s\n" % long_back_up_list[i])
                
        with open('latt_back_up.txt', 'a') as g:
            g.write("%s\n" % latt_back_up_list[i])

    df.to_csv(tweet_file, sep=',')

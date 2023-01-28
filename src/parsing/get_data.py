import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
import requests
import pandas as pd
import bs4
from bs4 import BeautifulSoup
from hyper.contrib import HTTP20Adapter
from hyper.http20.exceptions import StreamResetError
import time
import random
import os
from src.parsing.parsing_utils import read_json_params, rewrite_json_params


def get_with_retry(s, url, **params):
    try:
        r = s.get(url, params=params)
    except StreamResetError:
        time.sleep(5)
        r = s.get(url, params=params)
    return r

def get_hrefs(r):
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    divs = soup.find_all('div', class_="iva-item-titleStep-pdebR")
    hrefs = ['https://www.avito.ru' + div.find('a').get('href') for div in divs]
    return hrefs


def get_text(r):
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    div = soup.find('div', itemprop="description")
    texts = div
    if texts is None:
        return ''
    for text in texts:
        if type(text) == bs4.element.Tag:
            for e in text.find_all('br'):
                e.replace_with(' ')
            for e in text.find_all('p'):
                e.replace_with(' ')
    data = ' '.join([text.text for text in texts])
    return data


def get_number_of_pages(s, url):
    r = get_with_retry(s, url)
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    try:
        number_of_pages = soup.find('span', {'data-marker': 'pagination-button/next'}).previous_element
    except:
        number_of_pages = 1
    return int(number_of_pages)


DATA_PATH = '../../data/raw/texts_chunks'

texts = []
hrefs_list = []
texts_df = pd.DataFrame(columns=['texts'])
s = requests.Session()
s.mount('https://', HTTP20Adapter())

params = read_json_params()
query_id = params['query_id']
curr_url = params['current_url']
curr_page = params['current_page']
urls = ["https://www.avito.ru/moskva/kvartiry/sdam/na_dlitelnyy_srok-ASgBAgICAkSSA8gQ8AeQUg?cd=1&f=ASgBAQICAkSSA8gQ8AeQUgFAzAgkjFmOWQ",
        "https://www.avito.ru/moskva/kvartiry/sdam/na_dlitelnyy_srok/2-komnatnye-ASgBAQICAkSSA8gQ8AeQUgFAzAgUkFk?cd=1",
        "https://www.avito.ru/moskva/kvartiry/sdam/na_dlitelnyy_srok-ASgBAgICAkSSA8gQ8AeQUg?cd=1&f=ASgBAQICAkSSA8gQ8AeQUgFAzAgklFmSWQ"]
#url = "https://www.avito.ru/moskva/kvartiry/sdam/na_dlitelnyy_srok-ASgBAgICAkSSA8gQ8AeQUg?cd=1"
for url_num, url in enumerate(urls[curr_url:], start=curr_url):
    number_of_pages = get_number_of_pages(s, url)
    for page in range(curr_page, number_of_pages + 1):
        r = get_with_retry(s, url, page=page)
        hrefs = get_hrefs(r)
        for i, href in enumerate(hrefs):
            time.sleep(random.uniform(3, 7))
            print(url_num, page, i)
            print(href)
            ad_r = get_with_retry(s, href, page=page)
            hrefs_list.append(href)
            texts.append(get_text(ad_r))
            print(texts[-1])
        if page % 5 == 0 or page == number_of_pages:
            df = pd.DataFrame({'texts': texts, 'hrefs': hrefs_list})
            print(df)
            file_name = str(query_id) + '_texts_' + str(url_num) + '_' + str(page) + '.csv'
            output_file_path = os.path.join(DATA_PATH, file_name)
            df.to_csv(output_file_path, index=False)
            texts = []
            hrefs_list = []
            rewrite_json_params(current_url=url_num, current_page=page + 1)
        print(texts)
        print(len(texts))
    curr_page = 1
    rewrite_json_params(current_url=url_num + 1, current_page=curr_page)
rewrite_json_params(process_finished=1)

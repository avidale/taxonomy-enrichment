import requests
from bs4 import BeautifulSoup
import multiprocessing.pool
import time
import pickle
from collections import Counter
import argparse
import pandas as pd

import hashlib


def hash_float(text, salt='1', base=1000000):
    return int(hashlib.sha256((text+salt).encode('utf-8')).hexdigest(), 16) % base / base


STRESS_CHAR = '́'


def get_definition(query, mobile=True, retries=10, retry_timeout=3.0):
    print(query)
    if mobile:
        domain = 'ru.m.wikipedia.org'
    else:
        domain = 'ru.wikipedia.org'
    url = 'http://{}/w/index.php?search={}'.format(domain, query)
    for attempt in range(retries):
        try:
            t = requests.get(url).text
            break
        except Exception as e:
            print('Got error "{}" for query "{}" on attempt {}, will retry in {} seconds'.format(
                e, query, attempt, retry_timeout
            ))
            time.sleep(retry_timeout)
    soup = BeautifulSoup(t)
    d = soup.find('div', {'id': 'mw-content-text'}).find('div')
    if not d:
        return 'no_d'
    if mobile:
        sec = d.find('section')
        if sec:
            d = sec
    first_paragraph = 'no_p'
    for c in d.children:
        if c.name == 'p':
            first_paragraph = c.text
            break
    return first_paragraph.replace(STRESS_CHAR, '').replace('\xa0', ' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=761, type=int)
    parser.add_argument('--batch-size', default=1000, type=int)
    parser.add_argument('--threads', default=16, type=int)
    args = parser.parse_args()

    #public_test_nouns = pd.read_csv('../data/public_test/nouns_public.tsv', header=None, encoding='utf-8')
    #public_test_nouns.columns = ['noun']
    #nouns = public_test_nouns.noun

    train_n = pd.read_csv('../data/training_data/training_nouns.tsv', sep='\t', encoding='utf-8')
    train_n['one_text'] = train_n.TEXT.apply(lambda x: sorted(x.split(','), key=hash_float)[0])
    nouns = train_n.one_text
    #nouns = pd.read_pickle('ttest_dev.pkl').one_text

    # 8,  mobile: 16.1 / 100
    # 16, mobile: 11.8 / 100 (13.15 full)
    # 32, mobile: 10.3 / 100 (13.8 full)
    # 64, mobile:  9.7 / 100 (16.11 full)
    # 100,mobile: 9.9  / 100 or 48 / 500 (almost no scaling)
    batch_size = args.batch_size
    for batch_id in range(args.start, args.end):
        print('starting batch {}'.format(batch_id))
        first = batch_id * batch_size
        last = first + batch_size
        words = nouns.iloc[first:last].tolist()
        if not words:
            print('The doc is over!')
            break
        t0 = time.time()
        # results = [download_number(i) for i in range(100)]
        with multiprocessing.pool.ThreadPool(args.threads) as pool:  # vs pool.ThreadPool vs Pool
            docs = pool.map(get_definition, words)
        t1 = time.time()
        print('Batch {}.\ntime elapsed: {}'.format(batch_id, t1 - t0))
        with open('cache/wiki_train_{}_{}.pkl'.format(first, last), 'wb') as f:
            pickle.dump(docs, f)

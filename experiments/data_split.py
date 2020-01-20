import pandas as pd
import hashlib


def hash_float(text, salt='1', base=1000000):
    return int(hashlib.sha256((text+salt).encode('utf-8')).hexdigest(), 16) % base / base


def load_train_data(path='../data/training_data/training_nouns.tsv'):
    train_n = pd.read_csv(path, sep='\t', encoding='utf-8')
    train_n['parents_list'] = train_n.PARENTS.apply(lambda x: x.split(','))

    ttrain = train_n[train_n.synset_hash <= 0.8]
    ttest = train_n[train_n.synset_hash > 0.8]
    ttest_dev = ttest[ttest.synset_hash <= 0.82]
    ttest_test1 = ttest[(ttest.synset_hash > 0.82) & (ttest.synset_hash <= 0.84)]
    ttest_test2 = ttest[(ttest.synset_hash > 0.84) & (ttest.synset_hash <= 0.86)]
    ttest_hidden = ttest[(ttest.synset_hash > 0.86)]
    forbidden_id = set(ttest.SYNSET_ID)
    return ttrain, ttest_dev, ttest_test1, ttest_test2, ttest_hidden, forbidden_id


from collections import Counter, defaultdict
from sklearn.neighbors import KDTree

import numpy as np
import pandas as pd


from nltk import wordpunct_tokenize
from string import punctuation


from tqdm.auto import tqdm


def tokenize(text):
    return [t for t in wordpunct_tokenize(text.lower()) if not all(c in punctuation for c in t)]


def normalize(v):
    return v / sum(v**2)**0.5


class SentenceEmbedder:
    def __init__(self, ft, n=300):
        self.ft = ft
        self.n = n

    def __call__(self, text):
        tokens = tokenize(text)
        vecs = [normalize(self.ft[w]) for w in tokens]
        if len(vecs) == 0:
            return np.zeros(self.n)
        return normalize(sum(vecs))


class SynsetStorage:
    def __init__(self, id2synset, ids, ids_long, texts_long):
        self.id2synset = id2synset
        self.ids = ids
        self.ids_long = ids_long
        self.texts_long = texts_long

    @classmethod
    def construct(cls, synsets_raw):
        id2synset = {v['@id']: v for v in synsets_raw['synsets']['synset']}
        ids = sorted(id2synset.keys())
        ids_long = []
        texts_long = []
        for id in ids:
            s = id2synset[id]
            senses = s['sense']
            if not isinstance(senses, list):
                senses = [senses]
            texts = {sense['#text'] for sense in senses}
            texts.add(s['@ruthes_name'])
            if len(texts) > 1:
                texts.add(' ; '.join(sorted(texts)))
            for text in sorted(texts):
                ids_long.append(id)
                texts_long.append(text)
        print(len(ids), len(ids_long))
        return cls(
            id2synset=id2synset,
            ids=ids,
            ids_long=ids_long,
            texts_long=texts_long,
        )

# todo: implement this smarter way to construct index
"""
ids_long = []
texts_long = []
for id in ids:
    s = id2synset[id]
    senses = s['sense']
    if not isinstance(senses, list):
        senses = [senses]
    texts = {sense['#text'] for sense in senses}
    texts.add(s['@ruthes_name'])

    # исключаем все слова, омонимичные с тем, что есть в тестовой выборке
    senses = {synset_id for w in texts for synset_id in word2sense[w]}
    if senses.intersection(forbidden_id):
        continue

    if len(texts) > 1:
        texts.add(' ; '.join(sorted(texts)))
    for text in sorted(texts):
        ids_long.append(id)
        texts_long.append(text)
print(len(ids), len(ids_long))
"""


def make_rel_df(rel_n_raw, id2synset):
    rel_df = pd.DataFrame(rel_n_raw['relations']['relation'])
    rel_df['parent'] = rel_df['@parent_id'].apply(lambda x: id2synset[x]['@ruthes_name'])
    rel_df['child'] = rel_df['@child_id'].apply(lambda x: id2synset.get(x, {}).get('@ruthes_name'))
    return rel_df


class RelationStorage:
    def __init__(self, forbidden_id=None):
        self.id2hyponym = defaultdict(set)
        self.id2hypernym = defaultdict(set)
        self.forbidden_id = forbidden_id or set()  # forbidden_id = set(ttest.SYNSET_ID)

    def add_pair(self, hypo_id, hyper_id, max_depth=100500):
        if max_depth <= 0:
            return
        if hypo_id in self.id2hyponym[hyper_id]:
            # the pair is already here
            return
        if hypo_id in self.id2hypernym[hyper_id]:
            raise ValueError('{} is already a hypernym of {}, so it cannot become its hyponym'.format(hypo_id, hyper_id))
        for next_hypo in self.id2hyponym[hypo_id]:
            self.add_pair(next_hypo, hyper_id, max_depth=max_depth-1)
        for next_hyper in self.id2hypernym[hyper_id]:
            self.add_pair(hypo_id, next_hyper, max_depth=max_depth-1)
        self.id2hyponym[hyper_id].add(hypo_id)
        self.id2hypernym[hypo_id].add(hyper_id)

    def construct_relations(self, rel_df):
        self.id2hyponym = defaultdict(set)
        self.id2hypernym = defaultdict(set)

        hypo_df = rel_df[rel_df['@name'] == 'hyponym']
        for r, row in tqdm(hypo_df.iterrows()):
            hypo_id = row['@child_id']
            hyper_id = row['@parent_id']
            if hypo_id not in self.forbidden_id and hyper_id not in self.forbidden_id:
                self.add_pair(hypo_id, hyper_id, max_depth=1)  # во второй версии поставим максимальную глубину, равную 2

        print(len(self.id2hyponym))
        print(max(len(c) for c in self.id2hyponym.values()))
        print(max(len(c) for c in self.id2hypernym.values()))
        print(sum(len(c) for c in self.id2hypernym.values()))


def hypotheses_knn(text, index, text2vec, synset_storage: SynsetStorage, rel_storage: RelationStorage, k=10, verbose=False, decay=0, grand_mult=1):
    ids_list = synset_storage.ids_long
    texts_list = synset_storage.texts_long
    # todo: distance decay
    vec = text2vec(text)
    distances, indices = index.query(vec.reshape(1, -1), k=k)
    hypotheses = Counter()
    for i, d in zip(indices.ravel(), distances.ravel()):
        hypers = rel_storage.id2hypernym.get(ids_list[i], set())
        if verbose:
            print(d, 1, ids_list[i], texts_list[i], len(hypers))
        for parent in hypers:
            hypotheses[parent] += np.exp(-d**decay)
            for grandparent in rel_storage.id2hypernym.get(parent, set()):
                hypotheses[grandparent] += np.exp(-d**decay) * grand_mult
    if verbose:
        print(len(hypotheses))
    result = []
    for hypo, cnt in hypotheses.most_common(10):
        if verbose:
            print(cnt, hypo, synset_storage.id2synset[hypo]['@ruthes_name'])
        result.append(hypo)
    return result


def prepare_submission(words, hypotheses, id2synset):
    result_nouns = []
    result_hyperonyms = []
    result_hyper_names = []
    for n, h in zip(words, hypotheses):
        for hypo in h:
            result_nouns.append(n)
            result_hyperonyms.append(hypo)
            result_hyper_names.append(id2synset[hypo]['@ruthes_name'])
    result_df = pd.DataFrame({'noun': result_nouns, 'result': result_hyperonyms, 'result_text': result_hyper_names})
    return result_df


def dict2submission(word2hypotheses, id2synset):
    result_nouns = []
    result_hyperonyms = []
    result_hyper_names = []
    for n, h in word2hypotheses.items():
        for hypo in h:
            result_nouns.append(n)
            result_hyperonyms.append(hypo)
            result_hyper_names.append(id2synset[hypo]['@ruthes_name'])
    result_df = pd.DataFrame({'noun': result_nouns, 'result': result_hyperonyms, 'result_text': result_hyper_names})
    return result_df

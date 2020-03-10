import pandas as pd

from nlp import prepare_definition
from download_wiki import get_definition



def add_defin_hypotheses(text, hypotheses, definition_extractor, synset_storage, first_sentence=True, prefix='wiki'):
    defin_raw = definition_extractor.extract(text)
    defin = prepare_definition(defin_raw, first_sentence=first_sentence)
    for hn, h in hypotheses.items():
        for p in ['_min_place', '_match_len', '_n_senses', '_n_matches']:
            h[prefix + p] = 0
    for w, wsenses in synset_storage.word2sense.items():
        if w in defin:
            for sense in wsenses:
                h = hypotheses[sense]
                h['query'] = text
                h['document'] = sense
                h[prefix + '_min_place'] = min(defin.find(w), h.get(prefix + '_first_pos', 100500))
                h[prefix + '_match_len'] = len(w)  # todo: get maximum
                h[prefix + '_n_senses'] = len(wsenses)  # todo: get_mininum
                h[prefix + '_n_matches'] += 1
    return hypotheses

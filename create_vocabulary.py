import json

import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--language', required=True, help='ISO language code')
args = parser.parse_args()
relevant_columns = ['locale_dictionary', 'locale_fasttext', 'POS', 'rejected', 'reason']

language = args.language
token_dump_path = f'databases/{language}.csv'

vocab = {}
results = []
for path in tqdm(glob(f'json/{language}/vocab_{language}_*.json')):
    new_vocab = json.load(open(path, 'r'))
    for lemma, lemma_dict in new_vocab.items():
        for token, token_dict in lemma_dict['tokens'].items():
            results.append({
                'lemma': lemma,
                'token': token,
                'token_count': token_dict['count'],
                'article': path.split('_')[-1].split('.')[0],
                **{col: token_dict[col] for col in relevant_columns}
            })
        if lemma not in vocab:
            vocab[lemma] = lemma_dict
        else:
            vocab[lemma]['count'] += lemma_dict['count']
            for token_key in lemma_dict['tokens'].keys():
                if token_key in vocab[lemma]['tokens']:
                    vocab[lemma]['tokens'][token_key]['count'] += 1
                else:
                    vocab[lemma]['tokens'][token_key] = lemma_dict['tokens'][token_key]

results_df = pd.DataFrame(results)
results_df.to_csv(token_dump_path, index=False)

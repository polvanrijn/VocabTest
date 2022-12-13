import argparse
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import get_padding, expand_lemmas, compute_conditional_probabilities, compute_conditional_probability_from_n_gram

parser = argparse.ArgumentParser()
parser.add_argument('--language', required=True, help='ISO language code')
parser.add_argument('--n_gram_len', default=5, type=int, help='N in N-gram')
parser.add_argument('--n_pseudowords', default=500, type=int, help='Number of pseudoword pairs')
args = parser.parse_args()
language_iso = args.language
n_gram_len = args.n_gram_len
n_pseudowords = args.n_pseudowords

padding = get_padding(n_gram_len)

pseudowords = pd.read_csv(f'pseudowords/{language_iso}-pseudowords.csv')

noun_df = pd.read_csv(f'databases/{language_iso}-lemma_noun_df.csv')
noun_df, _ = expand_lemmas(noun_df, language_iso)
noun_df, n_gram_df = compute_conditional_probabilities(noun_df, padding)

probabilities = []
for n_gram_str in tqdm(pseudowords.n_grams, desc='Computing conditional probabilities'):
    n_grams = eval(n_gram_str)
    probabilities.append(compute_conditional_probability_from_n_gram(n_grams, n_gram_df))
pseudowords['probabilities'] = probabilities

def prepare_df(df):
    df['bits'] = df.apply(lambda row: -np.log2(row.probabilities), axis=1)
    df['num_n_grams'] = df.apply(lambda row: len(row.probabilities), axis=1)
    return df

pseudowords = prepare_df(pseudowords)
noun_df = prepare_df(noun_df)

available_pseudowords = pd.DataFrame()
for num_n_grams in set(pseudowords.num_n_grams):
    num_real_words = noun_df.query(f"num_n_grams == {num_n_grams}").shape[0]
    available_pseudowords = pd.concat([
        available_pseudowords,
        pseudowords.query(f"num_n_grams == {num_n_grams}").sort_values('match', ascending=False).tail(num_real_words)
    ])

if available_pseudowords.shape[0] >= n_pseudowords:
    available_pseudowords = available_pseudowords.sort_values('match', ascending=False).tail(n_pseudowords)
else:
    warnings.warn(f'Not enough pseudowords available for {language_iso} ({available_pseudowords.shape[0]} available, {n_pseudowords} requested)')


def add_to_pairs(row, bit_diff, pairs):
    if 'pseudoword' in row:
        stimulus = row.pseudoword
        correct_answer = 'incorrect'
    else:
        stimulus = row.lemma
        correct_answer = 'correct'
    pairs.append({
        'pair_id': len(pairs)//2 + 1,
        'bit_diff': bit_diff,
        'stimulus': stimulus,
        'correct_answer': correct_answer,
        'n_grams': row.n_grams,
        'match': row.get('match', None),
        'lemma_fpmw': row.get('lemma_fpmw', None),
        'lemma_zipf': row.get('lemma_zipf', None),
        'bits': row.bits,
        'num_n_grams': row.num_n_grams,
        'probability': row.probabilities,
    })
    return pairs

pairs = []
for num_n_grams in set(available_pseudowords.num_n_grams):
    selected_nouns = noun_df.query(f"num_n_grams == {num_n_grams}")
    selected_pseudowords = available_pseudowords.query(f"num_n_grams == {num_n_grams}")
    noun_bits = np.array(list(selected_nouns.bits))
    pseudo_bits = np.array(list(selected_pseudowords.bits))
    bit_diffs = []
    for pseudo_bit_array in pseudo_bits:
        bit_diffs.append(np.abs(noun_bits - pseudo_bit_array).mean(axis=1))
    bit_diffs = np.array(bit_diffs)
    block_idx = []
    for i in range(bit_diffs.shape[0]):
        minimal_bit_diff = bit_diffs.min(axis=1)
        pseudo_idx = np.argmin(minimal_bit_diff)
        noun_idx = np.argmin(bit_diffs[pseudo_idx])
        bit_diff = bit_diffs[pseudo_idx, noun_idx]

        pairs = add_to_pairs(selected_pseudowords.iloc[pseudo_idx], bit_diff, pairs)
        pairs = add_to_pairs(selected_nouns.iloc[noun_idx, :], bit_diff, pairs)
        bit_diffs[pseudo_idx, :] = np.inf
        bit_diffs[:, noun_idx] = np.inf

pairs_df = pd.DataFrame(pairs)
assert all(pairs_df.stimulus.value_counts() == 1)
pairs_df.to_csv(f'pairs/detailed/{language_iso}.csv', index=False)
pairs_df[['stimulus', 'correct_answer']].to_csv(f'pairs/minified/{language_iso}.csv', index=False)
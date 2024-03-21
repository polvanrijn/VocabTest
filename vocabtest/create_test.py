import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from vocabtest import package_dir
from vocabtest.utils.sampling import (
    compute_conditional_probability_from_n_gram, get_word_key
)
from vocabtest.utils.printing import warning


def _add_to_test(row, bit_diff, pairs, key):
    if 'pseudoword' in row:
        stimulus = row.pseudoword
        correct_answer = 'incorrect'
    else:
        stimulus = row[key]
        correct_answer = 'correct'
    pairs.append({
        'pair_id': len(pairs) // 2 + 1,
        'bit_diff': bit_diff,
        'stimulus': stimulus,
        'correct_answer': correct_answer,
        'n_grams': row.n_grams,
        'match': row.get('match', None),
        f'{key}_fpmw': row.get(f'{key}_fpmw', None),
        f'{key}_zipf': row.get(f'{key}_zipf', None),
        'bits': row.bits,
        'num_n_grams': row.num_n_grams,
        'probability': row.probabilities,
    })
    return pairs


def _prepare_df(df):
    df['bits'] = df.apply(lambda row: -np.log2(row.probabilities), axis=1)
    df['num_n_grams'] = df.apply(lambda row: len(row.probabilities), axis=1)
    return df

def create_test(dataset, language_iso, n_pseudowords, mean_value, std_value):
    database_folder = f'{package_dir}/{dataset}/databases'
    pseudoword_folder = f'{package_dir}/{dataset}/pseudowords'
    tests_folder = f'{package_dir}/{dataset}/tests'
    os.makedirs(tests_folder, exist_ok=True)
    os.makedirs(f"{tests_folder}/minified", exist_ok=True)
    os.makedirs(f"{tests_folder}/detailed", exist_ok=True)

    minified_path = f'{tests_folder}/minified/{language_iso}.csv'
    detailed_path = f'{tests_folder}/detailed/{language_iso}.csv'
    pseudowords = pd.read_csv(f'{pseudoword_folder}/{language_iso}-pseudowords.csv')
    if n_pseudowords > pseudowords.shape[0]:
        warning(f'Not enough pseudowords available for {language_iso} ({pseudowords.shape[0]} available, {n_pseudowords} requested)')
        n_pseudowords = pseudowords.shape[0]
    pseudowords['word_length'] = pseudowords.pseudoword.apply(len)

    filtered_word_df = pd.read_csv(f'{database_folder}/{language_iso}/{language_iso}-filtered.csv')
    n_gram_df = pd.read_csv(f'{database_folder}/{language_iso}/{language_iso}-n_gram_df.csv')
    if "word_length" not in filtered_word_df.columns:
        filtered_word_df['word_length'] = filtered_word_df.word.apply(len)
    # compute percentiles
    filtered_word_df['probability'] = filtered_word_df['count'] / filtered_word_df['count'].sum()
    epsilon = 1/(10 * filtered_word_df['count'].sum())
    filtered_word_df['epsilon_probability'] = filtered_word_df.probability.apply(lambda x: x + np.random.normal() * epsilon)
    filtered_word_df['log10_probability'] = np.log10(filtered_word_df.epsilon_probability)
    filtered_word_df.sort_values('log10_probability', ascending=False, inplace=True)
    filtered_word_df.reset_index(drop=True, inplace=True)

    key = get_word_key(filtered_word_df)

    if filtered_word_df.shape[0] < n_pseudowords:
        n_pseudowords = filtered_word_df.shape[0]
        selected_pseudowords = []
        for word_length, n_items in filtered_word_df["word_length"].value_counts().to_dict().items():
            available_pseudowords = pseudowords.query(f"word_length == {word_length}")
            if available_pseudowords.shape[0] < n_items:
                selected_pseudowords.extend(available_pseudowords.pseudoword.to_list())
            else:
                selected_pseudowords.extend(available_pseudowords.sample(n_items).pseudoword.to_list())
        # set seed
        np.random.seed(0)
        selected_pseudowords.extend(pseudowords.query("pseudoword not in @selected_pseudowords").sample(n_pseudowords - len(selected_pseudowords)).pseudoword.to_list())
        pairs_df = pd.concat([
            pd.DataFrame({
                'stimulus': filtered_word_df[key].to_list(),
                'correct_answer': True
            }),
            pd.DataFrame({
                'stimulus': selected_pseudowords,
                'correct_answer': False
            })
        ], axis=0)
        pairs_df[['stimulus', 'correct_answer']].to_csv(minified_path, index=False)

    # Half std_value to have 95% of the words within the mean_value +- std_value
    np.random.seed(0)
    possible_log_probabilities = np.random.normal(mean_value, std_value/2, n_pseudowords * 10)

    accepted_nouns = []

    for log_probability in tqdm(possible_log_probabilities):
        tmp_df = filtered_word_df.copy().query('word not in @accepted_nouns')
        tmp_df['diff'] = np.abs(tmp_df['log10_probability'] - log_probability)
        tmp_df = tmp_df.sort_values('diff')
        if len(tmp_df) == 0:
            break
        accepted_nouns.append(tmp_df[key].to_list()[0])
        if len(set(accepted_nouns)) == n_pseudowords * 10:
            break

    # Only keep nouns in the range
    filtered_word_df['word'] = filtered_word_df[key]
    filtered_word_df = filtered_word_df.query('word in @accepted_nouns').reset_index(drop=True)

    probabilities = []
    for n_gram_str in tqdm(pseudowords.n_grams, desc='Computing conditional probabilities of pseudowords'):
        n_grams = eval(n_gram_str)
        probabilities.append(compute_conditional_probability_from_n_gram(n_grams, n_gram_df))
    pseudowords['probabilities'] = probabilities


    pseudowords = _prepare_df(pseudowords)
    filtered_word_df['probabilities'] = filtered_word_df.apply(lambda row: eval(row.probabilities), axis=1)
    filtered_word_df = _prepare_df(filtered_word_df)

    available_pseudowords = pd.DataFrame()
    for num_n_grams in set(pseudowords.num_n_grams):
        num_real_words = filtered_word_df.query(f"num_n_grams == {num_n_grams}").shape[0]
        available_pseudowords = pd.concat([
            available_pseudowords,
            pseudowords.query(f"num_n_grams == {num_n_grams}").sort_values('match', ascending=False).tail(num_real_words)
        ])

    if available_pseudowords.shape[0] >= n_pseudowords:
        available_pseudowords = available_pseudowords.sort_values('match', ascending=False).tail(n_pseudowords)
    else:
        warning(f'Not enough pseudowords available for {language_iso} ({available_pseudowords.shape[0]} available, {n_pseudowords} requested)')



    pairs = []
    for num_n_grams in set(available_pseudowords.num_n_grams):
        selected_nouns = filtered_word_df.query(f"num_n_grams == {num_n_grams}")
        selected_pseudowords = available_pseudowords.query(f"num_n_grams == {num_n_grams}")
        noun_bits = np.array(list(selected_nouns.bits))
        pseudo_bits = np.array(list(selected_pseudowords.bits))
        bit_diffs = []
        for pseudo_bit_array in pseudo_bits:
            bit_diffs.append(np.abs(noun_bits - pseudo_bit_array).mean(axis=1))
        bit_diffs = np.array(bit_diffs)
        for i in range(bit_diffs.shape[0]):
            minimal_bit_diff = bit_diffs.min(axis=1)
            pseudo_idx = np.argmin(minimal_bit_diff)
            noun_idx = np.argmin(bit_diffs[pseudo_idx])
            bit_diff = bit_diffs[pseudo_idx, noun_idx]

            pairs = _add_to_test(selected_pseudowords.iloc[pseudo_idx], bit_diff, pairs, key)
            pairs = _add_to_test(selected_nouns.iloc[noun_idx, :], bit_diff, pairs, key)
            bit_diffs[pseudo_idx, :] = np.inf
            bit_diffs[:, noun_idx] = np.inf

    pairs_df = pd.DataFrame(pairs)
    assert all(pairs_df.stimulus.value_counts() == 1)
    pairs_df.head(n_pseudowords*2).to_csv(detailed_path, index=False)
    pairs_df[['stimulus', 'correct_answer']].to_csv(minified_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--language', required=True, help='ISO language code')
    parser.add_argument('--mean_value', default=-5, type=float, help='Mean percentile of words to include')
    parser.add_argument('--std_value', default=0.88, type=float, help='Std percentile of words to include')
    parser.add_argument('--n_pseudowords', default=500, type=int, help='Number of pseudoword pairs')
    args = parser.parse_args()

    create_test(args.dataset, args.language, args.n_pseudowords, args.mean_value, args.std_value)
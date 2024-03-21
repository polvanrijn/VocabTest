import subprocess

import tempfile

import re
import json
import requests

from os.path import exists

import pandas as pd
from tqdm import tqdm
from random import sample

from vocabtest import package_dir
from vocabtest.utils.converting import character_to_letter
from vocabtest.utils.validation import validate_pseudoword
from vocabtest.utils.iso import iso3_to_iso2
def get_padding(n_gram_len):
    """
    Get padding string for n-grams
    :param n_gram_len: length of n-grams
    :return: padding string, padding length
    """
    pad_len = n_gram_len - 1
    return '*' * (pad_len)

def get_word_key(word_df):
    selected_key = None
    for key in ['lemma', 'token', 'word']:
        if key in word_df.columns:
            selected_key = key
            break
    assert selected_key is not None
    return selected_key

def expand_word(word_df, language_iso, do_uromanize=False, key=None):
    if key is None:
        key = get_word_key(word_df)
    exp_key = key + '_expanded'
    word_df = word_df.iloc[[i for i, word in enumerate(word_df[key]) if type(word) == str], :]

    is_cjk = language_iso in ['zh', 'ja', 'ko']
    #do_uromanize = language_iso in ['kor', 'zho_tw', 'lzh', 'jpn', 'hak']
    primitives_to_character = {}
    if is_cjk:
        character2primitives = {}
        expanded_words = []
        for word in tqdm(word_df[key], desc='Expanding ' + key):
            expanded_word = ''
            for character in word:
                primitives = character_to_letter(character, language_iso)
                expanded_word += primitives
                if character not in character2primitives:
                    character2primitives[character] = primitives
            expanded_words.append(expanded_word)
        word_df[exp_key] = expanded_words
        reverse_primitives = {v: k for k, v in character2primitives.items()}
        for k in sorted(reverse_primitives, key=len, reverse=True):
            primitives_to_character[k] = reverse_primitives[k]
    elif do_uromanize:
        characters = set([character for word in word_df[key] for character in word])
        iso2_to_iso3 = dict(zip(iso3_to_iso2.values(), iso3_to_iso2.keys()))
        iso3 = iso2_to_iso3.get(language_iso, None)

        with tempfile.TemporaryDirectory() as tmp_folder:
            with open(f'{tmp_folder}/tmp.txt', 'w') as f:
                f.write('\n'.join(characters))
            cmd = f'perl {package_dir}/bible/dependencies/uroman/bin/uroman.pl '
            if iso3 is not None:
                cmd += f'-l {iso3} '
            cmd += f'< {tmp_folder}/tmp.txt'
            print(cmd)
            uromanized_characters = subprocess.check_output(cmd, shell=True).decode('utf-8').split('\n')[:-1]
        assert len(characters) == len(uromanized_characters)
        character2primitives = dict(zip(characters, uromanized_characters))
        expanded_words = []
        for word in tqdm(word_df[key], desc='Expanding ' + key):
            expanded_word = ''
            for character in word:
                primitives = character2primitives[character]
                expanded_word += primitives
            expanded_words.append(expanded_word)
        word_df[exp_key] = expanded_words
        reverse_primitives = {v: k for k, v in character2primitives.items()}
        for k in sorted(reverse_primitives, key=len, reverse=True):
            primitives_to_character[k] = reverse_primitives[k]
    else:
        word_df[exp_key] = word_df[key]

    word_df[exp_key + '_length'] = [len(word) for word in word_df[exp_key]]

    primitives_to_character = {
        key: value
        for key, value in primitives_to_character.items()
        if isinstance(value, str) and isinstance(key, str)
    }
    print(primitives_to_character)

    return word_df, primitives_to_character


conditional_probabilities = {}


def compute_conditional_probability_from_n_gram(n_grams, n_gram_df):
    """
    Compute conditional probabilities from n-grams for a given word
    :param n_grams: list of n-grams for a given word
    :param n_gram_df: table of all n-grams
    :return: conditional probabilities
    """
    pad_len = len(n_grams[0]) - 1
    probabilities = []
    for i in range(len(n_grams) - 1):
        n_gram = n_grams[i]
        next_n_gram = n_grams[i + 1]
        pair = (n_gram, next_n_gram)
        if pair not in conditional_probabilities:
            row_idx = [cur_gram.startswith(n_gram[1:]) for cur_gram in n_gram_df.n_gram]
            prob_df = n_gram_df.loc[row_idx, :]
            row_idx = prob_df.n_gram == next_n_gram
            conditional_probabilities[pair] = (prob_df['count'] / prob_df['count'].sum())[row_idx].values[0]
        probabilities.append(conditional_probabilities[pair])
    # we want to remove the last probabilities, because the probabilities will be one, e.g. at* will always have t** as
    # ending
    return probabilities[:-(pad_len - 1)]


def get_n_grams_from_word(word, padding):
    """
    Compute n-grams from a word with padding on both sides
    :param word: word to compute n-grams from
    :param padding: string to pad with, consisting only of asterisks (*), e.g. "***"
    :return: list of n-grams
    """
    padded_word = padding + word + padding
    n_gram_len = len(padding) + 1
    return [padded_word[i:i + n_gram_len] for i in range(len(word) + len(padding))]

def compute_conditional_probabilities(word_df, padding, key=None):
    if key is None:
        key = get_word_key(word_df)
    count_key = key + '_count'
    exp_key = key + '_expanded'

    idxs = word_df[exp_key].apply(lambda x: type(x) == str)
    word_df = word_df.loc[idxs, :]
    n_gram_list = []
    for bool_idx, row in word_df.iterrows():
        n_gram_list.extend(get_n_grams_from_word(row[exp_key], padding) * row[count_key])

    n_gram_df = pd.Series(n_gram_list).value_counts().reset_index(level=0)
    n_gram_df.columns = ['n_gram', 'count']

    probabilities = []
    n_gram_list = []
    for word in tqdm(word_df[exp_key], desc='Computing conditional probabilities'):
        n_grams = get_n_grams_from_word(word, padding)
        n_gram_list.append(n_grams)
        probabilities.append(compute_conditional_probability_from_n_gram(n_grams, n_gram_df))

    word_df['n_grams'] = n_gram_list
    word_df['num_n_grams'] = [len(n_grams) for n_grams in word_df['n_grams']]

    word_df['probabilities'] = probabilities
    word_df['num_probabilities'] = [len(probabilities) for probabilities in word_df['probabilities']]
    return word_df, n_gram_df

def sample_n_gram(start_key, n_gram_df):
    """
    Sample an possible n-gram given the last n-1 characters
    :param start_key: last n-1 characters
    :param n_gram_df: table of all n-grams
    :return: next n-gram
    """
    row_idx = [n_gram.startswith(start_key) for n_gram in n_gram_df.n_gram]
    df_selection = n_gram_df.loc[row_idx, :].copy()
    lst = [[row['n_gram']] * row['count'] for _, row in df_selection.iterrows()]
    flat_list = [item for sublist in lst for item in sublist]
    selection = sample(flat_list, 1)[0]
    return selection

def create_pseudoword(
    padding, n_gram_df, min_word_size, max_word_size, language_iso, approved_words, all_words, splitter, primitives_to_character
):
    """
    Create a pseudoword by sampling n-grams from the n-gram table
    :param padding: asterisks to pad with
    :param n_gram_df: table of all n-grams
    :param min_word_size: minimum word size
    :param max_word_size: maximum word size
    :param language_iso: language iso code
    :param all_words: set of all tokens and lemmas
    :param splitter: compound splitter
    :param primitives_to_character: only used if convert_chars is True, maps 'letters' to characters
    :return:
    """
    pad_len = len(padding)
    n_grams = []
    n_gram = sample_n_gram(padding, n_gram_df)
    n_grams.append(n_gram)
    word = n_gram
    while True:
        possible_next = sample_n_gram(word[-pad_len:], n_gram_df)
        n_grams.append(possible_next)
        letter = possible_next[-1]
        word += letter
        if possible_next.endswith(padding):
            word = word.replace('*', '')
            break
    is_valid = validate_pseudoword(
        pseudoword=word,
        min_word_size=min_word_size,
        max_word_size=max_word_size,
        language_iso=language_iso,
        approved_words=approved_words,
        all_words=all_words,
        splitter=splitter,
        primitives_to_character=primitives_to_character
    )
    return is_valid, n_grams


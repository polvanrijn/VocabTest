import json

import pandas as pd
import pykakasi  # Japanese
import jamotools  # Korean
from pypinyin import pinyin  # Chinese

import numpy as np
from tqdm import tqdm

from charsplit import Splitter
from random import sample
from fuzzywuzzy import fuzz


##################################################
# Helper functions
##################################################

def character_to_letter(character, language_iso):
    """
    Convert a CJK character to a 'letter', i.e., Japanese to hiragana, Chinese to pinyin, Korean to hangul
    :param character: character to convert
    :param language_iso: language iso code
    :return: converted character
    """
    if language_iso == 'ja':
        return syllabify_japanese(character, 'hiragana')
    elif language_iso == 'zh':
        return pinyin(character)[0][0]
    elif language_iso == 'ko':
        return ''.join(jamotools.split_syllables(character))
    else:
        raise ValueError(f'Language not supported: {language_iso}')


kks = pykakasi.kakasi()


def syllabify_japanese(word, type):
    """
    Convert a word to hiragana or katakana
    :param word: Japanese word to syllabify
    :param type: either 'katakana' or 'hiragana'
    :return: syllabified word
    """
    assert type in ['katakana', 'hiragana']
    key = 'kana' if type == 'katakana' else 'hira'
    return ''.join([w[key] for w in kks.convert(word)])

def get_padding(n_gram_len):
    """
    Get padding string for n-grams
    :param n_gram_len: length of n-grams
    :return: padding string, padding length
    """
    pad_len = n_gram_len - 1
    return '*' * (pad_len)

##################################################
# Conditional probabilities
##################################################

def expand_lemmas(noun_df, language_iso):
    is_cjk = language_iso in ['zh', 'ja', 'ko']
    primitives_to_character = {}
    if is_cjk:
        character2primitives = {}
        expanded_lemmas = []
        for lemma in tqdm(noun_df.lemma, desc='Expanding lemmas'):
            expanded_lemma = ''
            for character in lemma:
                primitives = character_to_letter(character, language_iso)
                expanded_lemma += primitives
                if character not in character2primitives:
                    character2primitives[character] = primitives
            expanded_lemmas.append(expanded_lemma)
        noun_df['lemma_expanded'] = expanded_lemmas
        reverse_primitives = {v: k for k, v in character2primitives.items()}
        for k in sorted(reverse_primitives, key=len, reverse=True):
            primitives_to_character[k] = reverse_primitives[k]
    else:
        noun_df['lemma_expanded'] = noun_df['lemma']

    noun_df['lemma_expanded_length'] = [len(lemma) for lemma in noun_df.lemma_expanded]
    return noun_df, primitives_to_character


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

def compute_conditional_probabilities(noun_df, padding):
    n_gram_list = []
    for bool_idx, row in noun_df.iterrows():
        n_gram_list.extend(get_n_grams_from_word(row['lemma_expanded'], padding) * row['lemma_count'])

    n_gram_df = pd.Series(n_gram_list).value_counts().reset_index(level=0)
    n_gram_df.columns = ['n_gram', 'count']

    probabilities = []
    n_gram_list = []
    for word in tqdm(noun_df.lemma_expanded, desc='Computing conditional probabilities'):
        n_grams = get_n_grams_from_word(word, padding)
        n_gram_list.append(n_grams)
        probabilities.append(compute_conditional_probability_from_n_gram(n_grams, n_gram_df))

    noun_df['n_grams'] = n_gram_list
    noun_df['num_n_grams'] = [len(n_grams) for n_grams in noun_df['n_grams']]

    noun_df['probabilities'] = probabilities
    noun_df['num_probabilities'] = [len(probabilities) for probabilities in noun_df['probabilities']]
    return noun_df, n_gram_df

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


##################################################
# Create and validate pseudowords
##################################################

def create_pseudoword(
        padding, n_gram_df, min_word_size, max_word_size, language_iso, block_set, lemma_set, primitives_to_character
):
    """
    Create a pseudoword by sampling n-grams from the n-gram table
    :param padding: asterisks to pad with
    :param n_gram_df: table of all n-grams
    :param min_word_size: minimum word size
    :param max_word_size: maximum word size
    :param language_iso: language iso code
    :param block_set: set of all tokens and lemmas
    :param lemma_set: set of all accepted lemmas
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
    return validate_pseudoword(word, min_word_size, max_word_size, language_iso, block_set, lemma_set,
                               primitives_to_character), n_grams


def validate_pseudoword(
        pseudoword, min_word_size, max_word_size, language_iso, block_set, lemma_set,
        primitives_to_character
):
    """
    Validate a pseudoword. Following criteria are checked:
    - word length: word must be between min_word_size and max_word_size and must contain at least two characters
    - word must not be a real word (i.e., not in block_set); for Japanese, we also check if the word exists as hiragana,
      the words in the block_set are already converted to hiragana
    - if it is a CJK word, we check if all 'letters' are successfully converted back to a character
      if not, we check if the word is a possible compound word
    :param pseudoword: pseudoword to validate
    :param min_word_size: minimum word size
    :param max_word_size: maximum word size
    :param language_iso: language iso code
    :param block_set: set of all tokens and lemmas
    :param lemma_set: set of all accepted lemmas
    :param primitives_to_character: only used if convert_chars is True, maps 'letters' to characters
    :return: (True, pseudoword) if pseudoword is valid, (False, "Error message") otherwise
    """
    is_cjk = language_iso in ['zh', 'ja', 'ko']
    if len(pseudoword) == 1:
        return False, f'Word is one char'
    if len(pseudoword) < min_word_size:
        return False, f'Word is too short'
    if len(pseudoword) > max_word_size:
        return False, f'Word is too long'
    if len(pseudoword) <= 1:
        return False, 'Word is too short'
    if pseudoword in block_set:
        return False, 'Word already in token/lemma list'
    if language_iso == 'ja':
        hiragana_word = syllabify_japanese(pseudoword, 'hiragana')
        if hiragana_word in block_set:
            return False, 'Word already in hiragana token/lemma list'
    if is_cjk:
        old_word = pseudoword
        for primitives, character in primitives_to_character.items():
            pseudoword = pseudoword.replace(primitives, character)
        n_letters_not_replaced = sum([c in old_word for c in pseudoword])
        if n_letters_not_replaced > 0:
            return False, 'Not all letters replaced'
    else:
        if is_compound_word(pseudoword, language_iso, lemma_set):
            return False, 'Compound word'
    return True, pseudoword


def is_compound_word(word, language_iso, lemma_set):
    """
    Estimate possible cuts in compound words using CharSplit (https://github.com/dtuggener/CharSplit). Cuts with a
    positive score are considered as possible cuts, see appendix of Tuggener, Don (2016). Incremental Coreference
    Resolution for German. For
    :param word:
    :param language_iso:
    :param lemma_set:
    :return: True if word is a compound word, False otherwise
    """
    splitter = Splitter(language_iso)
    cuts = splitter.split_compound(word)
    if len(cuts) > 1:
        if any([c[-1] in lemma_set for c in cuts if c[0] >= 0]):
            return True
    return False


def get_max_match(word, blocklist, fix_len=3):
    """
    Get the maximum match of a word to other words in a blocklist. This is used as a proxy for typos. To make the search
    more efficient, we only look at words that start with the same first and same last three letters and have a similar
    word length (1 letter tolerance on both sides per 10 letters). For speed, we vectorize the fuzzy match.
    :param word: word to check
    :param blocklist: list of words to check against
    :param fix_len: number of letters to check at the beginning and end of the word
    :return: maximum match score, None if the word is smaller than fix_len or if no match is found
    """
    word_len = len(word)
    if word_len < fix_len:
        return None
    char_allow = int(np.ceil(word_len / 10))
    min_word_len = word_len - char_allow
    max_word_len = word_len + char_allow
    prefix = word[:fix_len]
    suffix = word[-fix_len:]
    sublist = [
        block_item
        for block_item in blocklist
        if len(block_item) >= min_word_len and len(block_item) <= max_word_len
           and (block_item[:fix_len] == prefix or block_item[-fix_len:] == suffix)
    ]
    if len(sublist) == 0:
        return None
    return max(np.vectorize(fuzz.ratio)([word], sublist))


def create_block_set(language_iso):
    dump = pd.read_csv(f'databases/{language_iso}.csv')
    block_set = set(list(set(dump.token)) + list(set(dump.lemma)))
    block_set = {word for word in block_set if type(word) == str}
    with open(f'databases/{language_iso}-meta.json', 'r') as f:
        meta = json.load(f)

    if meta['lowercase_only']:
        block_set = {w.lower() for w in block_set}
    else:
        block_set = {w.upper() for w in block_set}

    if language_iso == 'ja':
        add_to_block_set = set()
        for word in tqdm(block_set, desc='Converting block_set to hiragana'):
            try:
                add_to_block_set.add(syllabify_japanese(word, 'hiragana'))
            except:
                pass
        block_set = block_set.union(add_to_block_set)
    return block_set

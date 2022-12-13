from collections import Counter
import pandas as pd
from tqdm import tqdm
import argparse
from utils import (
    get_padding,
    create_block_set,
    expand_lemmas,
    compute_conditional_probabilities,
    get_max_match,
    create_pseudoword,
)

parser = argparse.ArgumentParser()
parser.add_argument('--language', required=True, help='ISO language code')
parser.add_argument('--n_gram_len', default=5, type=int, help='N in N-gram')
args = parser.parse_args()

language_iso = args.language
n_gram_len = args.n_gram_len

# Defaults
MAX_TRIES = 10 ** 6
PSEUDO_WORDS = 1000
PADDING = get_padding(n_gram_len)

noun_df = pd.read_csv(f'databases/{language_iso}-lemma_noun_df.csv')
noun_df = noun_df.loc[[type(lemma) == str for lemma in noun_df.lemma], :]

lemma_df = pd.read_csv(f'databases/{language_iso}-lemma_df.csv')
lemma_df = lemma_df.loc[[type(lemma) == str for lemma in lemma_df.lemma], :]
lemma_set = set(lemma_df.query(f'POS=="NOUN" and accepted == True').lemma)

# Create the blocklist
block_set = create_block_set(language_iso)
block_list = list(block_set)

noun_df, primitives_to_character = expand_lemmas(noun_df, language_iso)

min_word_size = min(noun_df.lemma_expanded_length)
max_word_size = max(noun_df.lemma_expanded_length)

noun_df, n_gram_df = compute_conditional_probabilities(noun_df, PADDING)

messages = []
pbar = tqdm(range(MAX_TRIES))
pseudoword_results = []
pseudowords = []
for i in pbar:
    (success, pseudoword), n_grams = create_pseudoword(PADDING, n_gram_df, min_word_size, max_word_size, language_iso, block_set, lemma_set, primitives_to_character)
    if not success:
        messages.append(pseudoword)

    if pseudoword in pseudowords:
        messages.append('Pseudoword already exists')
        success = False

    if success:
        pseudowords.append(pseudoword)
        pseudoword_results.append({
            'pseudoword': pseudoword,
            'n_grams': n_grams,
            'num_n_grams': len(n_grams),
            'match': get_max_match(pseudoword, block_list), # Check for possible typos
        })
        messages.append('Success')
        pd.DataFrame(pseudoword_results).to_csv(f'pseudowords/{language_iso}-pseudowords.csv', index=False)
    pbar.set_description(str(dict(Counter(messages))))
    if len(pseudoword_results) >= PSEUDO_WORDS:
        break
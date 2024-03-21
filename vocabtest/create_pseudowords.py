import os
os.environ['HOME'] = '/home/pol.van-rijn'
import time
from collections import Counter
from os.path import exists

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
from vocabtest import package_dir
from vocabtest.utils.sampling import (
    get_padding,
    expand_word,
    get_word_key,
    compute_conditional_probabilities,
    create_pseudoword,
)
from vocabtest.utils.validation import get_max_match
from vocabtest.utils.charsplit import Splitter

def plot_history(history, language_iso, dataset):
    history.to_csv(f"{package_dir}/{dataset}/pseudowords/{language_iso}-history.csv", index=False)
    df = pd.DataFrame(history, columns=['message', 'date', 'count']) \
        .set_index('date') \
        .sort_values('date')

    last_date = df.index.max()
    n_pseudowords = 0
    if 'Success' in df.message.tolist():
        success = df.loc[last_date].query("message == 'Success'")['count'].values
        if len(success) > 0:
            n_pseudowords = success[0]


    duration = df.index.max() - df.index.min()
    if duration.days > 0:
        duration_str = f"{duration.days} days"
    else:
        duration = duration.seconds / 60
        duration_str = f"{duration:.2f} min"

    duration_title = f"{language_iso}: {duration_str}, n = {n_pseudowords} ({dataset})"

    pt = pd.pivot_table(df, columns=['message'], index=['date'], values=['count'], fill_value=0)
    pt.columns = pt.columns.droplevel()
    pt.plot.area()
    plt.title(duration_title)
    plt.savefig(f"{package_dir}/{dataset}/pseudowords/{language_iso}.png")
    plt.close()



def create_pseudowords(dataset, language_iso, n_gram_len, max_tries, n_pseudowords, verbose=False):
    database_folder = f'{package_dir}/{dataset}/databases'
    pseudoword_folder = f'{package_dir}/{dataset}/pseudowords'
    os.makedirs(pseudoword_folder, exist_ok=True)
    PADDING = get_padding(n_gram_len)

    filtered_word_df = pd.read_csv(f'{database_folder}/{language_iso}/{language_iso}-filtered.csv')
    filtered_word_df = filtered_word_df.loc[[type(word) == str for word in filtered_word_df.word], :]

    with open(f'{database_folder}/{language_iso}/{language_iso}-clean.txt', 'r') as f:
        approved_words = set(f.read().splitlines())

    # Create the blocklist
    with open(f'{database_folder}/{language_iso}/{language_iso}-all.txt', 'r') as f:
        all_words = set(f.read().splitlines())
    all_words_list = list(all_words)

    primitives_path = f'{database_folder}/{language_iso}/{language_iso}-primitives.csv'
    key = get_word_key(filtered_word_df)
    primitives_to_character = {}
    if f'{key}_expanded_length' not in filtered_word_df.columns:
        # run pseudoword generation on all accepted words, not only in the small frequency range
        filtered_word_df, primitives_to_character = expand_word(filtered_word_df, language_iso, key=key)
        if primitives_to_character != {}:
            primitives_df = pd.DataFrame({
                'primitive': list(primitives_to_character.keys()),
                'character': list(primitives_to_character.values())
            })
            primitives_df.to_csv(primitives_path, index=False)


    min_word_size = min(filtered_word_df[f'{key}_expanded_length'])
    max_word_size = max(filtered_word_df[f'{key}_expanded_length'])

    if exists(primitives_path):
        primitives_df = pd.read_csv(primitives_path).dropna()
        primitives_to_character = dict(zip(primitives_df.primitive, primitives_df.character))



    if 'n_grams' in filtered_word_df.columns:
        n_gram_df = pd.read_csv(f'{database_folder}/{language_iso}/{language_iso}-n_gram_df.csv')

    else:
        filtered_word_df, n_gram_df = compute_conditional_probabilities(filtered_word_df, PADDING, key=key)
        n_gram_df.to_csv(f'{database_folder}/{language_iso}/{language_iso}-n_gram_df.csv', index=False)
        filtered_word_df.to_csv(f'{database_folder}/{language_iso}/{language_iso}-filtered.csv', index=False)

    messages = []
    pseudoword_results = []
    pseudowords = []
    splitter = Splitter(language_iso, dataset) if primitives_to_character == {} else None
    # failures = []

    start = time.time()
    history = pd.DataFrame()

    if verbose:
        pbar = tqdm(range(max_tries))
    else:
        pbar = range(max_tries)

    for i in pbar:
        (success, pseudoword), n_grams = create_pseudoword(
            padding=PADDING,
            n_gram_df=n_gram_df,
            min_word_size=min_word_size,
            max_word_size=max_word_size,
            language_iso=language_iso,
            approved_words=approved_words,
            all_words=all_words,
            splitter=splitter,
            primitives_to_character=primitives_to_character
        )
        if not success:
            messages.append(pseudoword)

        if pseudoword in pseudowords:
            messages.append('Already generated')
            success = False

        if success:
            pseudowords.append(pseudoword)
            pseudoword_results.append({
                'pseudoword': pseudoword,
                'n_grams': n_grams,
                'num_n_grams': len(n_grams),
                'match': get_max_match(pseudoword, all_words_list), # Check for possible typos
            })
            messages.append('Success')
            pd.DataFrame(pseudoword_results).to_csv(f'{pseudoword_folder}/{language_iso}-pseudowords.csv', index=False)

        # if not success:
        #     pseudoword = ''.join([n_gram[-1] for n_gram in n_grams]).replace('*', '')
        #     failures.append({
        #         'word': pseudoword,
        #         'reason': messages[-1]
        #     })

        # if 1 minute past
        if i + 1 % 100 == 0 or time.time() - start > 10:
            start = time.time()
            item = pd.Series(messages).value_counts().reset_index()
            item.columns = ['message', 'count']
            item['date'] = pd.Timestamp.now()
            history = pd.concat([history, item])
            plot_history(history, language_iso, dataset)
            # pd.DataFrame(failures).to_csv(f'{pseudoword_folder}/{language_iso}-failures.csv', index=False)
        if verbose:
            pbar.set_description(str(dict(Counter(messages))))

        # if i + 1 % 100 == 0:
        #     print(f'Generated {i + 1} pseudowords')
        #pbar.set_description(str(dict(Counter(messages))))
        if len(pseudoword_results) >= n_pseudowords:
            item = pd.Series(messages).value_counts().reset_index()
            item.columns = ['message', 'count']
            item['date'] = pd.Timestamp.now()
            history = pd.concat([history, item])
            plot_history(history, language_iso, dataset)
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--language', required=True, help='ISO language code')
    parser.add_argument('--n_gram_len', default=5, type=int, help='N in N-gram')
    parser.add_argument('--max_tries', default=10 ** 6, type=int, help='Maximum number of tries')
    parser.add_argument('--n_pseudowords', default=1000, type=int, help='Number of pseudowords to generate')
    parser.add_argument('--verbose', default=False, type=int, help='Print progress')
    args = parser.parse_args()


    create_pseudowords(args.dataset, args.language, args.n_gram_len, args.max_tries, args.n_pseudowords, args.verbose)
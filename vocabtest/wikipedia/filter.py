import json
import argparse

import numpy as np  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from vocabtest import package_dir
from vocabtest.utils.charsplit import train_splitter, Splitter
from vocabtest.utils.checks import basic_checks, lang_2_script, get_char_df, in_accepted_script, is_compound_word
from vocabtest.utils.converting import syllabify_japanese
from vocabtest.utils.printing import warning, info, print_summary

MINIMAL_WORDS_PER_MILION = 5
DATASET = 'wikipedia'




def get_all_words(dump, token_df, language_iso):
    all_words = set(
        list(set(dump.token)) + list(set(dump.lemma)) +
        list(set(token_df.token)) + list(set(token_df.lemma))
    )
    all_words = {word for word in all_words if type(word) == str}

    # Make sure all entries are lowercase
    all_words = {w.lower() for w in all_words}

    # Also add hiragana versions of all words for Japanese, because Japanese in contrast to Chinese and Korean is
    # written in mixed scripts (kanji, hiragana, katakana)
    if language_iso == 'ja':
        converted_words = set()
        for word in tqdm(all_words, desc='Converting all_words to hiragana'):
            try:
                converted_words.add(syllabify_japanese(word, 'hiragana'))
            except:
                pass
        all_words = all_words.union(converted_words)

    return all_words


def count_valid_tokens(df):
    return df.query('rejected == False').token_count.sum()


MILION = 10 ** 6


def check_enough_valid_tokens(df):
    return count_valid_tokens(df) / MILION < MINIMAL_WORDS_PER_MILION


def filter(language_iso):
    """
    Process the language dump (`{language_iso}.csv`) to create the following files:
    - `{language_iso}-filtered.csv` is a table with `word` and `count` of all words that pass the filter,
    - `{language_iso}-clean.txt` is txt file with all words that are cleaned -> used for training the compound word splitter,
    - `{language_iso}-all.txt` is txt file with all words occurring in the corpus -> used to reject pseudowords which are already in the corpus
    :param language_iso:
    :return:
    """
    accepted_scripts = lang_2_script[language_iso]

    meta = {
        'warnings': []
    }

    databases_folder = f'{package_dir}/{DATASET}/databases'

    token_article_dump = pd.read_csv(f'{databases_folder}/{language_iso}/{language_iso}.csv')
    null_idx = token_article_dump.reason.isnull()
    token_article_dump.loc[null_idx, 'reason'] = 'no reason'  # To avoid errors when grouping

    # Remove empty tokens
    na_idx = token_article_dump.token.isna()
    meta['number_na_token_removed'] = int(na_idx.sum())
    print_summary(f'Removing NaN tokens', na_idx.sum(), token_article_dump.shape[0])
    token_article_dump = token_article_dump.loc[[not b for b in na_idx], :]
    total_articles = len(set(token_article_dump['article']))
    expected_articles = 10000
    if total_articles < expected_articles:
        meta['warnings'].append('too_few_articles')
        print_summary('Not enough articles', total_articles, expected_articles, type='warning')

    meta['total_token_count'] = int(token_article_dump.token_count.sum())

    # Strip quotes
    for str_col in ['lemma', 'token']:
        token_article_dump[str_col] = token_article_dump[str_col].str.replace('"', '')

    # Remove separation chars in lemma
    for sep_char in ['_', '#']:
        token_article_dump['lemma'] = token_article_dump['lemma'].str.replace(sep_char, '')

    # Some languages contain punctuation in regular words, so we will not reject them
    if language_iso in ['sa', 'ta', 'te', 'hi', 'mr', 'vi']:
        warning('Regular words contains punctuation')
        meta['warnings'].append('punctuation_in_regular_words')
        idx = token_article_dump.reason == 'contains_non_alphanumeric'
        token_article_dump.loc[idx, 'rejected'] = False

    # Loosen language conditions: accept if lexicon or fasttext detects it
    locale_or_idx = \
        (token_article_dump.reason == "wrong_locale") & \
        ((token_article_dump.locale_dictionary == language_iso) | (token_article_dump.locale_fasttext == language_iso))
    token_article_dump.loc[locale_or_idx, 'rejected'] = False

    print_summary(f'Rejected tokens', token_article_dump.rejected.sum(), token_article_dump.shape[0])

    # Make sure you have at least 5 million valid tokens
    if check_enough_valid_tokens(token_article_dump):
        if locale_or_idx.sum() / len(token_article_dump) <= 0.35:
            warning('Locale not reliably detected, therefore accepting all tokens with wrong locale.')
            token_article_dump.loc[token_article_dump['reason'] == 'wrong_locale', 'rejected'] = False
            meta['warnings'].append('locale_not_detected')
        if check_enough_valid_tokens(token_article_dump):
            msg = f'We recommend to have at least {MINIMAL_WORDS_PER_MILION} million valid tokens, you only have'
            n_valid_tokens = count_valid_tokens(token_article_dump)
            print_summary(msg, n_valid_tokens, MINIMAL_WORDS_PER_MILION * MILION, 'warning')
            meta['warnings'].append('too_few_valid_tokens')

    # Remove tokens that are not in the accepted script
    rejected_idx = token_article_dump.rejected == False
    unrejected_token_article_dump = token_article_dump.loc[rejected_idx, :]

    n_chars, char_df, unicode_block_table = get_char_df(unrejected_token_article_dump)
    print_summary(f'Unique characters', char_df.shape[0], n_chars)
    print(unicode_block_table)

    if accepted_scripts is not None:
        new_rejections = [not in_accepted_script(token, accepted_scripts, char_df) for token in
                          unrejected_token_article_dump.token]
        token_article_dump.loc[rejected_idx, 'rejected'] = new_rejections
        token_article_dump.loc[rejected_idx, :].loc[new_rejections, 'reason'] = 'wrong_script'

    # Print lemma coverage
    no_lemma = sum((token_article_dump.lemma == '-') | (token_article_dump.lemma == 'unknown'))
    meta['lemma_coverage'] = no_lemma / token_article_dump.shape[0]
    print_summary('Tokens without lemma', no_lemma, token_article_dump.shape[0])

    assert meta['lemma_coverage'] < 0.5, 'Lemma coverage is too low'

    # Create token_df
    grouped_dump = token_article_dump.groupby(['token', 'lemma', 'POS', 'reason', 'rejected'])
    token_df = grouped_dump.token_count.sum().reset_index()
    token_df['n_articles'] = grouped_dump.article.count().reset_index().article
    assert all(token_df.token_count >= token_df.n_articles), "You cannot have more token counts than articles"

    # Lowercase all strings
    token_df.loc[:, 'token'] = [token.lower() for token in token_df.token]
    token_df.loc[:, 'lemma'] = [lemma.lower() for lemma in token_df.lemma]

    n_valid_tokens = token_df.query('rejected==False').shape[0]
    print_summary(f'Valid tokens of all tokens for {language_iso}', n_valid_tokens, token_df.shape[0])

    n_valid_nouns = token_df.query('rejected==False and POS=="NOUN"').shape[0]
    print_summary(f'Valid nouns of all valid tokens for {language_iso}', n_valid_nouns, n_valid_tokens)

    token_df['token_article_ratio'] = token_df.token_count / token_df.n_articles
    percentile = np.percentile(token_df.token_article_ratio, 95)
    above_idx = token_df.token_article_ratio > percentile

    token_df.loc[above_idx, 'rejected'] = True
    token_df.loc[above_idx, 'reason'] = token_df.loc[above_idx, 'reason'] + ', limited_number_articles'
    correlation = token_df.loc[above_idx, ['token_count', 'n_articles']].corr().iloc[1, 0]
    token_df['accepted'] = [not b for b in token_df.rejected]

    print_summary('95 percentile of token/article ratio', sum(above_idx), token_df.shape[0])

    lemma_df = token_df.groupby(
        ['lemma', 'POS']
    ).agg(
        reason=('reason', set),
        accepted=('accepted', any),
        lemma_count=('token_count', sum),
        tokens=('token', set),
        n_tokens=('token', 'count'),
    ).reset_index()

    str_idx = [type(lemma) == str for lemma in lemma_df.lemma]
    lemma_df = lemma_df.loc[str_idx, :]

    # exclude lemmas that do not pass basic checks, e.g. contain punctuation
    bad_lemma_bool_idx = []
    for lemma in lemma_df['lemma']:
        is_failure, _ = basic_checks(lemma)
        bad_lemma_bool_idx.append(is_failure)
    lemma_df.loc[bad_lemma_bool_idx, 'accepted'] = False

    # exclude lemmas that do not appear as tokens
    noun_df = token_df.query(f'POS == "NOUN" and rejected == False')
    all_tokens = set(noun_df.token.tolist())
    misspelled_lemma_bool_idx = [l not in all_tokens for l in lemma_df.lemma]
    lemma_df.loc[misspelled_lemma_bool_idx, 'accepted'] = False

    print_summary('Valid lemmas', lemma_df.query('accepted == True').shape[0], lemma_df.shape[0])

    total_token_count = token_article_dump.token_count.sum()
    total_lemma_count = lemma_df.lemma_count.sum()
    if total_token_count != total_lemma_count:
        warning(
            f'Warning: the sum of the token counts ({total_token_count}) is not equal to the sum of the lemma counts ({total_lemma_count})')

    lemma_df['lemma_fpmw'] = (lemma_df['lemma_count'] / lemma_df['lemma_count'].sum()) * MILION
    lemma_df['lemma_zipf'] = np.log10(lemma_df['lemma_fpmw']) + 3
    lemma_df['lemma_length'] = [len(lemma) for lemma in lemma_df.lemma]

    pipeline_steps = {
        'total lemmas': lemma_df.shape[0],
        'valid lemmas': lemma_df.query('accepted == True').shape[0],
        'valid nouns': lemma_df.query('accepted == True and POS=="NOUN"').shape[0],
    }

    valid_nouns = lemma_df.query('POS=="NOUN" and accepted==True')
    print_summary(f'Valid nouns for {language_iso}', valid_nouns.shape[0], lemma_df.shape[0])

    filtered_word_df = lemma_df.query(f'POS=="NOUN" and accepted == True and lemma_length > 1').copy()

    # Detect compound words
    if language_iso not in ['ja', 'ko', 'zh']:
        accepted_nouns = lemma_df.query(f'POS=="NOUN" and accepted == True')
        noun_tuples = [tuple(l) for l in accepted_nouns[['lemma', 'lemma_count']].values]
        train_splitter(noun_tuples, language_iso, DATASET)

        splitter = Splitter(language_iso, DATASET)
        approved_words = set(accepted_nouns.lemma)

        is_compound = []
        for lemma in tqdm(filtered_word_df.lemma):
            if type(lemma) != str:
                continue
            is_compound.append(is_compound_word(
                word=lemma,
                splitter=splitter,
                word_list=approved_words,
            ))
        accepted_lemmas = [not c for c in is_compound]
        meta['removed_compound_words'] = sum(is_compound)
        print_summary('Number of compound words found', sum(is_compound), len(is_compound))
        filtered_word_df = filtered_word_df.loc[accepted_lemmas, :]
        pipeline_steps['after removing\ncompound words'] = sum(accepted_lemmas)

    # Only keep nouns with typical length
    median_lemma_length = filtered_word_df.lemma_length.median()
    std_lemma_length = filtered_word_df.lemma_length.std()
    lemma_length_range = [
        int(max(1, np.floor(median_lemma_length - 2 * std_lemma_length))),
        int(np.ceil(median_lemma_length + 2 * std_lemma_length))
    ]
    filtered_word_df.query(
        f'lemma_length >= {lemma_length_range[0]} and lemma_length <= {lemma_length_range[1]}',
        inplace=True
    )
    filtered_word_df['word'] = filtered_word_df['lemma']
    filtered_word_df['count'] = filtered_word_df['lemma_count']
    pipeline_steps[f'nouns with\nnormal length {lemma_length_range}'] = filtered_word_df.shape[0]

    all_words = get_all_words(token_article_dump, token_df, language_iso)
    with open(f'{databases_folder}/{language_iso}/{language_iso}-all.txt', 'w') as f:
        f.write('\n'.join(all_words))

    lemma_df = lemma_df.loc[[type(lemma) == str for lemma in lemma_df.lemma], :]
    approved_words = set(lemma_df.query(f'POS=="NOUN" and accepted == True').lemma)

    with open(f'{databases_folder}/{language_iso}/{language_iso}-clean.txt', 'w') as f:
        f.write('\n'.join(approved_words))

    filtered_word_df.to_csv(f'{databases_folder}/{language_iso}/{language_iso}-filtered.csv', index=False)

    warning_msg = ''
    if len(meta["warnings"]) > 0:
        warning_msg = f'\n⚠: {meta["warnings"]}'
    fig, axis = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Diagnostic plot for {language_iso} {warning_msg}')
    token_df.token_article_ratio.hist(bins=2000, log=True, ax=axis[0, 0])
    axis[0, 0].axvline(percentile, color='red')
    axis[0, 0].set_xlabel('raw token count')
    axis[0, 0].set_ylabel('log article count')
    axis[0, 0].set_title(f'95 percentile, r={correlation:.2f}')

    # Zipf plot
    axis[0, 1].plot(lemma_df['lemma_zipf'].sort_values().values)
    axis[0, 1].set_title('Lemma zipf')

    axis[1, 0].bar(pipeline_steps.keys(), pipeline_steps.values())
    axis[1, 0].set_xticks(range(len(pipeline_steps.keys())))
    axis[1, 0].set_xticklabels(pipeline_steps.keys(), rotation=90, ha='right')
    axis[1, 0].set_ylabel(f'Number of lemmas ({language_iso})')
    axis[1, 0].set_title(f'Final number of nouns {filtered_word_df.shape[0]}')

    axis[1, 1] = filtered_word_df.lemma_length.value_counts().sort_index().plot.bar()
    axis[1, 1].set_title('Word distribution')
    plt.tight_layout()
    fig.savefig(f'{databases_folder}/{language_iso}/{language_iso}-diagnostic.png')

    meta['pipeline_steps'] = pipeline_steps
    with open(f'{databases_folder}/{language_iso}/{language_iso}-meta.json', 'w') as f:
        json.dump(meta, f)

    # For plotting example distribution
    # below_idx = token_df.token_article_ratio <= percentile
    # x1 = token_df.loc[above_idx, 'token_count']
    # y1 = token_df.loc[above_idx, 'n_articles']
    # x2 = token_df.loc[below_idx, 'token_count']
    # y2 = token_df.loc[below_idx, 'n_articles']
    # SIZE = 3
    # plt.cla()
    # plt.scatter(x1, y1, alpha=0.2, rasterized=True)
    # plt.scatter(x2, y2, alpha=0.2, rasterized=True)
    # plt.xlim(-99, 1500)
    # plt.ylim(-33, 500)
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('/tmp/above_below.pdf', dpi=400, width=3, height=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', required=True, help='ISO language code')
    args = parser.parse_args()
    filter(args.language)

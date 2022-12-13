import json
import re
import warnings
import requests
import argparse

import numpy as np  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
from charsplit import Splitter

from charsplit import train_splitter
from utils import is_compound_word

parser = argparse.ArgumentParser()
parser.add_argument('--language_iso', required=True, help='ISO language code')
args = parser.parse_args()

language_iso = args.language_iso
arabic = ['Arabic']
armenian = ['Armenian']
chinese = ['CJK Unified Ideographs']
cyrillic = ['Cyrillic', 'Cyrillic Supplement', 'Cyrillic Extended-A', 'Cyrillic Extended-B']
devanagari = ['Devanagari']
gothic = ['Gothic']
greek = ['Greek and Coptic', 'Greek Extended']
hebrew = ['Hebrew']
#japanese = ['Hiragana', 'Katakana', 'CJK Unified Ideographs']
# Japanese allows to mix characters with Hiragana and Katakana, however it is non-trivial to get a realistic mix of them
# in the pseudowords. We therefore only allow CJK ideographs, following the same procedure as for Chinese.
japanese = ['CJK Unified Ideographs']
korean = ['Hangul Syllables']
tamil = ['Tamil']
telugu = ['Telugu']
basic_latin = ['Basic Latin']
latin_1_supplement = basic_latin + ['Latin-1 Supplement']
latin_extended_a = latin_1_supplement + ['Latin Extended-A']
latin_extended_a_b = latin_extended_a + ['Latin Extended-B']
latin_extended_additional = latin_extended_a_b + ['Latin Extended Additional']  # Vietnamese

lang_2_script = {
    'af': latin_1_supplement,
    'ar': arabic,
    'be': cyrillic,
    'bg': cyrillic,
    'ca': latin_1_supplement,
    'cs': latin_extended_a,
    'cy': latin_extended_a,
    'da': latin_1_supplement,
    'de': latin_1_supplement,
    'el': greek,
    'en': basic_latin,
    'es': latin_1_supplement,
    'et': latin_extended_a,
    'eu': latin_1_supplement,
    'fa': arabic,
    'fi': latin_1_supplement,
    'fo': latin_1_supplement,
    'fr': latin_1_supplement,
    'ga': latin_1_supplement,
    'gd': latin_1_supplement,
    'gl': latin_1_supplement,
    'got': gothic,
    'he': hebrew,
    'hi': devanagari,
    'hr': latin_extended_a,
    'hu': latin_extended_a,
    'hy': armenian,
    'hyw': armenian,
    'id': basic_latin,
    'is': latin_1_supplement,
    'it': latin_1_supplement,
    'ja': japanese,
    'ko': korean,
    'la': latin_1_supplement,
    'lt': latin_1_supplement,
    'lv': latin_1_supplement,
    'mr': devanagari,
    'mt': latin_extended_a,
    'nl': latin_1_supplement,
    'nn': latin_1_supplement,
    'no': latin_1_supplement,
    'pl': latin_extended_a,
    'pt': latin_1_supplement,
    'ro': latin_extended_a_b,
    'ru': cyrillic,
    'sa': devanagari,
    'se': latin_extended_a,
    'sk': latin_extended_a,
    'sl': latin_1_supplement,
    'sr': cyrillic,
    'sv': latin_1_supplement,
    'ta': tamil,
    'te': telugu,
    'tr': latin_extended_a,
    'ug': arabic,
    'uk': cyrillic,
    'ur': arabic,
    'vi': latin_extended_additional,
    'wo': latin_1_supplement,
    'zh': chinese,
}

accepted_scripts = lang_2_script[language_iso]


ZIPF_RANGES = [3, 5]
MINIMAL_WORDS_PER_MILION = 5
meta = {
    'warnings': []
}

unicode_blocks = []
pattern = re.compile(r'([0-9A-F]+)\.\.([0-9A-F]+);\ (\S.*\S)')
response = requests.get('http://unicode.org/Public/UNIDATA/Blocks.txt')

for line in response.text.splitlines():
    m = pattern.match(line)
    if m:
      start, end, name = m.groups()
      unicode_blocks.append((int(start, 16), int(end, 16), name))

def block(ch):
    """
    Return the Unicode block name for ch, or None if ch has no block.
    """
    assert isinstance(ch, str) and len(ch) == 1, repr(ch)
    cp = ord(ch)
    for start, end, name in unicode_blocks:
        if start <= cp <= end:
          return name

def in_accepted_script(token):
    return all([char2block[c] in accepted_scripts for c in token])


def print_summary(msg, count, total, type='print'):
    full_message = f'{msg}: {count} ({count/total*100:.1f}%)'
    if type == 'print':
        print(full_message)
    elif type == 'warning':
        warnings.warn(full_message)
    else:
        raise NotImplementedError()

def count_valid_tokens(df):
    return df.query('rejected == False').token_count.sum()

def check_enough_valid_tokens(df):
    return count_valid_tokens(df) / 10 ** 6 < MINIMAL_WORDS_PER_MILION


token_article_dump = pd.read_csv(f'databases/{language_iso}.csv')
null_idx = token_article_dump.reason.isnull()
token_article_dump.loc[null_idx, 'reason'] = 'no reason'  # To avoid errors when grouping

# Remove empty tokens
na_idx = token_article_dump.token.isna()
meta['number_na_token_removed'] = int(na_idx.sum())
print_summary(f'Removing NaN tokens', na_idx.sum(), token_article_dump.shape[0])
token_article_dump = token_article_dump.loc[[not b for b in na_idx], :]
total_articles = len(set(token_article_dump['article']))
if total_articles < 10000:
    meta['warnings'].append('too_few_articles')
    print_summary('Not enough articles', total_articles, 10000, type='warning')

meta['total_token_count'] = int(token_article_dump.token_count.sum())

# Strip quotes
for str_col in ['lemma', 'token']:
    token_article_dump[str_col] = token_article_dump[str_col].str.replace('"', '')

# Remove separation chars in lemma
for sep_char in ['_', '#']:
    token_article_dump['lemma'] = token_article_dump['lemma'].str.replace(sep_char, '')

# Some languages contain punctuation in regular words, so we will not reject them
if language_iso in ['sa', 'ta', 'te', 'hi', 'mr', 'vi']:
    warnings.warn('Regular words contains punctuation')
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
        warnings.warn(f'Last resort! Locale not reliably detected for {language_iso}, therefore accepting all tokens with wrong locale.')
        token_article_dump.loc[token_article_dump['reason'] == 'wrong_locale', 'rejected'] = False
        n_valid_tokens = count_valid_tokens(token_article_dump)
        meta['warnings'].append('locale_not_detected')
    if check_enough_valid_tokens(token_article_dump):
        msg = f'We recommend to have at least {MINIMAL_WORDS_PER_MILION} million valid tokens, you only have'
        n_valid_tokens = count_valid_tokens(token_article_dump)
        print_summary(msg, n_valid_tokens, MINIMAL_WORDS_PER_MILION * 10 ** 6, 'warning')
        meta['warnings'].append('too_few_valid_tokens')

# Remove tokens that are not in the accepted script
rejected_idx = token_article_dump.rejected == False
unrejected_token_article_dump = token_article_dump.loc[rejected_idx, :]
all_chars = ''.join([row['token'] * row['token_count'] for _, row in unrejected_token_article_dump.iterrows()])
char_array = [c for c in all_chars]
char_df = pd.Series(char_array).value_counts().reset_index()
char_df.columns = ['char', 'count']
char_df['block'] = char_df.char.apply(block)
print_summary(f'Unique characters', char_df.shape[0], len(char_array))
print(char_df.groupby('block')['count'].sum().sort_values(ascending=False))

if accepted_scripts is not None:
    char2block = dict(zip(char_df.char, char_df.block))
    new_rejections = [not in_accepted_script(token) for token in unrejected_token_article_dump.token]
    token_article_dump.loc[rejected_idx, 'rejected'] = new_rejections
    token_article_dump.loc[rejected_idx, :].loc[new_rejections, 'reason'] = 'wrong_script'


# Print lemma coverage
no_lemma = sum((token_article_dump.lemma == '-') | (token_article_dump.lemma == 'unknown'))
meta['lemma_coverage'] = no_lemma/token_article_dump.shape[0]
print_summary('Tokens without lemma', no_lemma, token_article_dump.shape[0])

assert meta['lemma_coverage'] < 0.5, 'Lemma coverage is too low'

# Create token_df
grouped_dump = token_article_dump.groupby(['token', 'lemma', 'POS', 'reason', 'rejected'])
token_df = grouped_dump.token_count.sum().reset_index()
token_df['n_articles'] = grouped_dump.article.count().reset_index().article
assert all(token_df.token_count >= token_df.n_articles), "You cannot have more token counts than articles"


# Determine Noun capitalization
noun_df = token_df.query(f'POS == "NOUN" and rejected == False')
capitalized_nouns = sum([row['token'][0].isupper() * row['token_count'] for idx, row in noun_df.iterrows()])
total_nouns = sum(noun_df.token_count)
percentage_first_word_cap = capitalized_nouns / total_nouns
lowercase_only = percentage_first_word_cap < 0.5

if lowercase_only:
    print_summary('Lower-casing all tokens', capitalized_nouns, total_nouns)
else:
    print_summary('Upper-casing all tokens', capitalized_nouns, total_nouns)

meta['lowercase_only'] = lowercase_only

if lowercase_only:
    token_df.loc[:, 'token'] = [token.lower() for token in token_df.token]
    token_df.loc[:, 'lemma'] = [lemma.lower() for lemma in token_df.lemma]
else:
    token_df.loc[:, 'token'] = [token.upper() for token in token_df.token]
    token_df.loc[:, 'lemma'] = [lemma.upper() for lemma in token_df.lemma]


n_valid_tokens = token_df.query('rejected==False').shape[0]
print_summary(f'Valid tokens of all tokens for {language_iso}', n_valid_tokens, token_df.shape[0])

n_valid_nouns = token_df.query('rejected==False and POS=="NOUN"').shape[0]
print_summary(f'Valid nouns of all valid tokens for {language_iso}', n_valid_nouns, n_valid_tokens)

token_df.token_article_ratio = token_df.token_count/token_df.n_articles
percentile = np.percentile(token_df.token_article_ratio, 95)
above_idx = token_df.token_article_ratio > percentile

token_df.loc[above_idx, 'rejected'] = True
token_df.loc[above_idx, 'reason'] = token_df.loc[above_idx, 'reason'] + ', limited_number_articles'
correlation = token_df.loc[above_idx, ['token_count', 'n_articles']].corr().iloc[1,0]
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

PUNCTUATION = ['|', '\\', '^', ',', '(', '*', '"', '!', '$', '/', '[', '`', ';', ']', '#', '}', '&', '=', "'", '@', '~',
               '{', '>', '<', '%', '_', '?', '+', '-', ')', ':', '.']
contains_punctuation = []
for lemma in lemma_df['lemma']:
    contains_punctuation.append(any([l in PUNCTUATION for l in lemma]))
lemma_df.loc[contains_punctuation, 'accepted'] = False

print_summary('Valid lemmas', lemma_df.query('accepted == True').shape[0], lemma_df.shape[0])

total_token_count = token_article_dump.token_count.sum()
total_lemma_count = lemma_df.lemma_count.sum()
if total_token_count != total_lemma_count:
    warnings.warn(f'Warning: the sum of the token counts ({total_token_count}) is not equal to the sum of the lemma counts ({total_lemma_count})')

lemma_df['lemma_fpmw'] = (lemma_df['lemma_count'] / lemma_df['lemma_count'].sum()) * 10 ** 6
lemma_df['lemma_zipf'] = np.log10(lemma_df['lemma_fpmw']) + 3
lemma_df['lemma_length'] = [len(lemma) for lemma in lemma_df.lemma]

pipeline_steps = {
    'total lemmas': lemma_df.shape[0],
    'valid lemmas': lemma_df.query('accepted == True').shape[0],
    'valid nouns': lemma_df.query('accepted == True and POS=="NOUN"').shape[0],
}

valid_nouns = lemma_df.query('POS=="NOUN" and accepted==True')
print_summary(f'Valid nouns for {language_iso}', valid_nouns.shape[0], lemma_df.shape[0])

lemma_noun_df = lemma_df.query(f'POS=="NOUN" and accepted == True and lemma_zipf > {ZIPF_RANGES[0]} and lemma_zipf <= {ZIPF_RANGES[1]} and lemma_length > 1').copy()
print_summary(f'Nouns that lie in {ZIPF_RANGES} for {language_iso}', noun_df.shape[0], valid_nouns.shape[0])

pipeline_steps[f'nouns in {ZIPF_RANGES}'] = lemma_noun_df.shape[0]

# Detect compound words
if language_iso not in ['ja', 'ko', 'zh']:
    accepted_nouns = lemma_df.query(f'POS=="NOUN" and accepted == True')
    noun_tuples = [tuple(l) for l in accepted_nouns[['lemma', 'lemma_count']].values]
    train_splitter(noun_tuples, language_iso)

    splitter = Splitter(language_iso)
    lemma_set = set(accepted_nouns.lemma)

    is_compound = []
    for lemma in lemma_noun_df.lemma:
        if type(lemma) != str:
            continue
        is_compound.append(is_compound_word(lemma, language_iso, lemma_set))
    accepted_lemmas = [not c for c in is_compound]
    meta['removed_compound_words'] = sum(is_compound)
    print_summary('Number of compound words found', sum(is_compound), len(is_compound))
    lemma_noun_df = lemma_noun_df.loc[accepted_lemmas, :]
    pipeline_steps['after removing\ncompound words'] = sum(accepted_lemmas)

# Only keep nouns with typical length
median_lemma_length = lemma_noun_df.lemma_length.median()
std_lemma_length = lemma_noun_df.lemma_length.std()
lemma_length_range = [
    int(max(0, np.floor(median_lemma_length - 2 * std_lemma_length))),
    int(np.ceil(median_lemma_length + 2 * std_lemma_length))
]
lemma_noun_df.query(f'lemma_length >= {lemma_length_range[0]} and lemma_length <= {lemma_length_range[1]}', inplace=True)
pipeline_steps[f'nouns with\nnormal length {lemma_length_range}'] = lemma_noun_df.shape[0]

lemma_df.to_csv(f'databases/{language_iso}-lemma_df.csv', index=False)
lemma_noun_df.to_csv(f'databases/{language_iso}-lemma_noun_df.csv', index=False)
token_df.to_csv(f'databases/{language_iso}-token_df.csv', index=False)

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
axis[0, 1].hlines(ZIPF_RANGES[0], 0, lemma_df.shape[0], colors='r', linestyles='dashed')
axis[0, 1].hlines(ZIPF_RANGES[1], 0, lemma_df.shape[0], colors='r', linestyles='dashed')
axis[0, 1].set_title('Lemma zipf')

axis[1, 0].bar(pipeline_steps.keys(), pipeline_steps.values())
axis[1, 0].set_xticklabels(pipeline_steps.keys(), rotation=90, ha='right')
axis[1, 0].set_ylabel(f'Number of lemmas ({language_iso})')
axis[1, 0].set_title(f'Final number of nouns {lemma_noun_df.shape[0]}')

axis[1, 1] = lemma_noun_df.lemma_length.value_counts().sort_index().plot.bar()
axis[1, 1].set_title('Word distribution')
fig.savefig(f'databases/{language_iso}-diagnostic.png')

meta['pipeline_steps'] = pipeline_steps
with open(f'databases/{language_iso}-meta.json', 'w') as f:
    json.dump(meta, f)
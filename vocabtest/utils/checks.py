import os

import pandas as pd
import re
import requests

PUNCTUATION = [
    '|', '\\', '^', ',', '(', '*', '"', '!', '$', '/', '[', '`', ';', ']', '#', '}', '&', '=', "'", '@', '~',
    '{', '>', '<', '%', '_', '?', '+', '-', ')', ':', '.'
]


def basic_checks(word):
    # Reject words in all uppercase - they are probably acronyms
    if all([l.isupper() for l in word]):
        return True, 'all_upper'

    # Reject words with digits
    if any([l.isdigit() for l in word]):
        return True, 'contains_digit'

    # Reject words with punctuation
    if any([l in PUNCTUATION for l in word]):
        return True, 'contains_punctuation'

    # Reject single non-alphabetic characters
    if len(word) == 1 and bool(re.search(r'[a-z]', word)):
        return True, 'single_letter'

    return False, None




##################################################
# Unicode functions
##################################################
arabic = ['Arabic']
armenian = ['Armenian']
chinese = ['CJK Unified Ideographs']
cyrillic = ['Cyrillic', 'Cyrillic Supplement', 'Cyrillic Extended-A', 'Cyrillic Extended-B']
devanagari = ['Devanagari']
gothic = ['Gothic']
greek = ['Greek and Coptic', 'Greek Extended']
hebrew = ['Hebrew']
japanese = ['Hiragana', 'Katakana', 'CJK Unified Ideographs']
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


def get_char_df(token_count_df):
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

    all_chars = ''.join([row['token'] * row['token_count'] for _, row in token_count_df.iterrows()])
    char_array = [c for c in all_chars]
    char_df = pd.Series(char_array).value_counts().reset_index()
    char_df.columns = ['char', 'count']
    char_df['block'] = char_df.char.apply(block)

    unicode_block_table = char_df.groupby(['char', 'block'])['count'].sum().groupby('block').sum().sort_values(ascending=False).reset_index()
    return len(char_array), char_df, unicode_block_table


def in_accepted_script(token, accepted_scripts, char_df):
    char2block = dict(zip(char_df.char, char_df.block))
    return all([char2block[c] in accepted_scripts for c in token])

##################################################
# Compound word functions
##################################################

def is_compound_word(word, splitter, word_list):
    """
    Estimate possible cuts in compound words using CharSplit (https://github.com/dtuggener/CharSplit). Cuts with a
    positive score are considered as possible cuts, see appendix of Tuggener, Don (2016). Incremental Conference
    Resolution for German.
    :param word: word to be checked
    :param splitter: splitter
    :param word_list: all approved words
    :return: True if word is a compound word, False otherwise
    """
    cuts = splitter.split_compound(word)
    if len(cuts) > 1:
        if any([c[-1] in word_list for c in cuts if c[0] >= 0]):
            return True
    return False
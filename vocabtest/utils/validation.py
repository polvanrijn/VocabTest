import numpy as np


from vocabtest.utils.checks import is_compound_word
from vocabtest.utils.converting import syllabify_japanese
def validate_pseudoword(
        pseudoword, min_word_size, max_word_size, language_iso, all_words, approved_words, splitter,
        primitives_to_character
):
    """
    Validate a pseudoword. Following criteria are checked:
    - word length: word must be between min_word_size and max_word_size and must contain at least two characters
    - word must not be a real word (i.e., not in all_words); for Japanese, we also check if the word exists as hiragana,
      the words in the all_words are already converted to hiragana
    - if it is a CJK word, we check if all 'letters' are successfully converted back to a character
      if not, we check if the word is a possible compound word
    :param pseudoword: pseudoword to validate
    :param min_word_size: minimum word size
    :param max_word_size: maximum word size
    :param language_iso: language iso code
    :param all_words: set of all tokens and lemmas
    :param splitter: compound splitter
    :param primitives_to_character: only used if convert_chars is True, maps 'letters' to characters
    :return: (True, pseudoword) if pseudoword is valid, (False, "Error message") otherwise
    """
    if len(pseudoword) == 1:
        return False, f'One char'
    if len(pseudoword) < min_word_size:
        return False, f'Too short'
    if len(pseudoword) > max_word_size:
        return False, f'Too long'
    if len(pseudoword) <= 1:
        return False, 'Empty'
    if pseudoword in all_words:
        return False, 'In vocabulary'
    if language_iso == 'ja':
        hiragana_word = syllabify_japanese(pseudoword, 'hiragana')
        if hiragana_word in all_words:
            return False, 'In vocabulary'
    if primitives_to_character != {}:
        old_word = pseudoword
        for primitives, character in primitives_to_character.items():
            pseudoword = pseudoword.replace(primitives, character)
        n_letters_not_replaced = sum([c not in primitives_to_character.values() for c in pseudoword])
        if n_letters_not_replaced > 0:
            return False, 'Not all replaced'
        if pseudoword in all_words:
            return False, 'In vocabulary'
    else:
        if is_compound_word(word=pseudoword, splitter=splitter, word_list=approved_words):
            return False, 'Compound word'
    return True, pseudoword




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
    from fuzzywuzzy import fuzz

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


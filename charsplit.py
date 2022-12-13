"""
Modified from https://github.com/dtuggener/CharSplit
"""

import re
import sys
import json

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

def train_splitter(noun_tuples, language_iso, max_words: int = 10000000, max_len: int = 20) -> None:
    """
    Calculate ngram probabilities at different positions
    :param max_words: Max. no. of words to analyse
    :param max_len: Max. ngram length
    :return: None
    """
    assert language_iso not in ['ko', 'ja', 'zh'], 'Not implemented for CJK languages'
    print(f'Loaded {len(noun_tuples)} nouns {language_iso}')

    # Dicts for counting the ngrams
    end_ngrams = defaultdict(int)
    start_ngrams = defaultdict(int)
    in_ngrams = defaultdict(int)
    all_ngrams = defaultdict(int)

    # Gather counts
    print('Words analyzed of max.', str(max_words))
    c = 0   # Line counter

    for noun, count in tqdm(noun_tuples):
        line = noun.strip()

        if '-' in line:
            line = re.sub('.*-', '', line)  # Hyphen: take part following last hyphen

        line_middle = line[1:-1]

        for n in range(3, max_len+1):   # "Overcount" long words
        # for n in range(3, len(line)+1):   # Lower performance

            if n <= max_len:
                ngram = line[:n]      # start_grams: max_len 3-5
                start_ngrams[ngram] += count
                all_ngrams[ngram] += count

                ngram = line[-n:]     # end_grams: max_len 3-5
                end_ngrams[ngram] += count
                all_ngrams[ngram] += count

            for m in range(len(line_middle) - n + 1):   # in_grams: max_len 3-5
                ngram = line_middle[m:m+n]
                if not ngram == '':
                    in_ngrams[ngram] += count
                    all_ngrams[ngram] += count

        if c % 10000 == 0:
            sys.stderr.write('\r'+str(c))
            sys.stderr.flush()
        c += 1

        if c == max_words:
            break

    sys.stderr.write('\n')

    print('Calculating ngrams probabilities')
    start_ngrams = {k: v/all_ngrams[k] for k,v in start_ngrams.items() if v > 1}
    end_ngrams = {k: v/all_ngrams[k] for k,v in end_ngrams.items() if v > 1}
    in_ngrams = {k: v/all_ngrams[k] for k,v in in_ngrams.items() if v > 1}

    # Write dicts to file
    with open(str(Path(__file__).parent) + f'/{language_iso}_ngram_probs.json', "w") as f:
        json.dump({
            "prefix": start_ngrams,
            "infix": in_ngrams,
            "suffix": end_ngrams
        }, f)

class Splitter:
    def __init__(self, language_iso):
        ngram_path = Path(__file__).parent / f"{language_iso}_ngram_probs.json"
        with open(ngram_path) as f:
            self.ngram_probs = json.load(f)
        self.language_iso = language_iso

    """
    Wrapper around the split_compound function
    """
    def split_compound(self, word: str) -> List[Tuple[float, str, str]]:
        """Return list of possible splits, best first.
        :param word: Word to be split
        :return: List of all splits
        """
        word = word

        # If there is a hyphen in the word, return part of the word behind the last hyphen
        if '-' in word:
            return [(1., re.search('(.*)-', word).group(1), re.sub('.*-', '', word))]

        scores = list() # Score for each possible split position

        # Iterate through characters, start at forth character, go to 3rd last
        for n in range(3, len(word)-2):
            pre_slice = word[:n]

            # Cut of Fugen-S
            if pre_slice.endswith('ts') or pre_slice.endswith('gs') or pre_slice.endswith('ks') \
                    or pre_slice.endswith('hls') or pre_slice.endswith('ns'):
                if len(word[:n-1]) > 2: pre_slice = word[:n-1]

            # Start, in, and end probabilities
            pre_slice_prob = list()
            in_slice_prob = list()
            start_slice_prob = list()

            # Extract all ngrams
            for k in range(len(word)+1, 2, -1):

                # Probability of first compound, given by its ending prob
                if not pre_slice_prob and k <= len(pre_slice):
                    # The line above deviates from the description in the thesis;
                    # it only considers word[:n] as the pre_slice.
                    # This improves accuracy on GermEval and increases speed.
                    # Use the line below to replicate the original implementation:
                    # if k <= len(pre_slice):
                    end_ngram = pre_slice[-k:]  # Look backwards
                    pre_slice_prob.append(self.ngram_probs["suffix"].get(end_ngram, -1))   # Punish unlikely pre_slice end_ngram

                # Probability of ngram in word, if high, split unlikely
                in_ngram = word[n:n+k]
                in_slice_prob.append(self.ngram_probs["infix"].get(in_ngram, 1)) # Favor ngrams not occurring within words

                # Probability of word starting
                # The condition below deviates from the description in the thesis (see above comments);
                # Remove the condition to restore the original implementation.
                if not start_slice_prob:
                    ngram = word[n:n+k]
                    # Cut Fugen-S
                    if ngram.endswith('ts') or ngram.endswith('gs') or ngram.endswith('ks') \
                            or ngram.endswith('hls') or ngram.endswith('ns'):
                        if len(ngram[:-1]) > 2:
                            ngram = ngram[:-1]

                    start_slice_prob.append(self.ngram_probs["prefix"].get(ngram, -1))

            if not pre_slice_prob or not start_slice_prob:
                continue

            start_slice_prob = max(start_slice_prob)
            pre_slice_prob = max(pre_slice_prob)  # Highest, best pre_slice
            in_slice_prob = min(in_slice_prob)  # Lowest, punish splitting of good in_grams
            score = start_slice_prob - in_slice_prob + pre_slice_prob
            scores.append((score, word[:n], word[n:]))

        scores.sort(reverse=True)

        if not scores:
            scores = [[0, word, word]]

        return sorted(scores, reverse = True)
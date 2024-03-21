import numpy as np
import tempfile

import subprocess

import os
import re
import json

from glob import glob
import argparse
import pandas as pd
from thefuzz import fuzz
from tqdm import tqdm
from vocabtest import package_dir

from vocabtest.utils.charsplit import train_splitter, Splitter
from vocabtest.utils.checks import basic_checks, lang_2_script, get_char_df, in_accepted_script, is_compound_word
from vocabtest.utils.iso import iso3_to_iso2
from vocabtest.utils.printing import print_summary, info
from vocabtest.utils.sampling import expand_word

DATASET = 'bible'
data_dir =  f'{package_dir}/{DATASET}/data'
database_dir = f'{package_dir}/{DATASET}/databases'
dependencies_dir = f'{package_dir}/{DATASET}/dependencies'

fast_align_dir = f'{dependencies_dir}/fast_align'
theographic_dir = f'{dependencies_dir}/theographic-bible-metadata'

assert os.path.exists(fast_align_dir), f'Please clone https://github.com/clab/fast_align/ in {fast_align_dir} and build it'
assert os.path.exists(theographic_dir), f'Please clone https://github.com/robertrouse/theographic-bible-metadata in {theographic_dir}'

REFERENCE_BIBLE_PATH = f'{data_dir}/eng.json'
STOPWORD_PATH = f'{data_dir}/eng_stopwords.txt'

def pos_reference():
    import spacy

    names = (
        pd.read_csv(f'{theographic_dir}/CSV/Books.csv').bookName.tolist() +
        pd.read_csv(f'{theographic_dir}/CSV/People.csv').displayTitle.tolist() +
        pd.read_csv(f'{theographic_dir}/CSV/PeopleGroups.csv').groupName.tolist() +
        pd.read_csv(f'{theographic_dir}/CSV/Places.csv').displayTitle.tolist()
    )
    stopwords = set([re.sub("[\(\[].*?[\)\]]", "", name.lower()).strip() for name in names if type(name) == str])
    nlp = spacy.load("en_core_web_trf")

    prop_nouns = []
    for stopword in tqdm(stopwords, total=len(stopwords), desc='Subselecting proper nouns'):
        for token in nlp(stopword):
            if token.pos_ == 'PROPN':
                prop_nouns.append(token.text)
    stopwords = set([word for word in prop_nouns if word.isalnum()])

    nlp = spacy.load("en_core_web_sm")

    bible_names = sorted(glob(f'{data_dir}/eng/*.json'))

    bible = {}

    for bible_name in bible_names:
        with open(bible_name) as f:
            bible = {
                **bible,
                **json.load(f)
            }

    with open(REFERENCE_BIBLE_PATH, 'w') as f:
        json.dump(bible, f)


    summary = []
    for verse_id, verse in tqdm(bible.items()):
        for idx, token in enumerate(nlp(verse)):
            if token.pos_ == 'PROPN':
                # block_idx[verse_id].append(idx)
                summary.append({'word': token.text, 'reason': 'POS'})
            elif token.text in stopwords:
                # block_idx[verse_id].append(idx)
                summary.append({'word': token.text, 'reason': 'STOPWORD'})

    nlp = spacy.load("en_core_web_trf")
    extended_stop_list = []
    for word in tqdm(pd.DataFrame(summary).word.unique()):
        if nlp(word)[0].pos_ == 'PROPN':
            extended_stop_list.append(word)

    all_words = [word for verse in bible.values() for word in verse.split(' ')]
    token_df = pd.Series(all_words).value_counts().reset_index()
    token_df.columns = ['token', 'token_count']
    prop_noun_df = token_df[token_df.token.isin(extended_stop_list)]
    print('Top prop nouns')
    print(prop_noun_df.head(20))
    percentage = token_df[token_df.token.isin(extended_stop_list)].token_count.sum() / token_df.token_count.sum()
    print(f'Prop nouns make up {percentage:.2f}% of the corpus')

    with open(STOPWORD_PATH, 'w') as f:
        f.write('\n'.join(extended_stop_list))


def align_verses_with_reference(reference_bible, verse_dict):
    reference_list = []
    target_list = []
    verse_ids = []
    for verse_id, verse in tqdm(verse_dict.items()):
        if verse_id in reference_bible:
            reference_verse = reference_bible[verse_id]
            reference_list.append(reference_verse + '\n')
            target_list.append(verse + '\n')
            verse_ids.append(verse_id)
    return verse_ids, reference_list, target_list

def find_aligned_stop_words(all_verses, primitives_to_character):
    if not os.path.exists(STOPWORD_PATH):
        print('Computing block list from reference')
        pos_reference()

    with open(REFERENCE_BIBLE_PATH) as f:
        reference_bible = json.load(f)

    with open(STOPWORD_PATH) as f:
        stopwords = f.read().split('\n')

    verse_ids, reference_list, target_list = align_verses_with_reference(reference_bible, all_verses)

    # Dump the aligned verses to a file
    with tempfile.TemporaryDirectory() as alignment_folder:
        ref_path = f'{alignment_folder}/reference.txt'
        with open(ref_path, 'w') as reference_file:
            reference_file.writelines(reference_list)

        target_path = f'{alignment_folder}/target.txt'
        with open(target_path, 'w') as target_file:
            target_file.writelines(target_list)

        lines = []
        for i in range(len(target_list)):
            ref = reference_list[i].replace('\n', '')
            trg = target_list[i].replace('\n', '')
            lines.append(f'{ref} ||| {trg}\n')
        os.makedirs(alignment_folder, exist_ok=True)
        aligned_text_file = f'{alignment_folder}/test.txt'
        with open(aligned_text_file, 'w') as f:
            f.writelines(lines)

        fast_align = f'{fast_align_dir}/build/fast_align'
        alignment_file = f'{alignment_folder}/alignment.txt'
        print('Aligning source and target')
        subprocess.call(f"{fast_align} -i {aligned_text_file} -d -o -v > {alignment_file}", shell=True)

        with open(alignment_file, 'r') as f:
            alignment = f.readlines()

        alignment_summary = []
        for line_idx, line in tqdm(enumerate(alignment), total=len(alignment), desc='Parsing alignment'):
            for idx in line.replace('\n', '').split(' '):
                if idx == '':
                    continue
                src, trg = idx.split('-')
                src_idx = int(src)
                trg_idx = int(trg)

                src_split = reference_list[line_idx].replace('\n', '').split(' ')
                trg_split = target_list[line_idx].replace('\n', '').split(' ')
                src_word = src_split[src_idx]
                trg_word = trg_split[trg_idx]

                alignment_summary.append({
                    'src': src_word,
                    'trg': trg_word,
                    'stopword': src_word in stopwords
                })
        alignment_df = pd.DataFrame(alignment_summary)
        if len(alignment_df) == 0:
            return []
        src_word_count = alignment_df.src.value_counts()
        frequent_words = src_word_count[src_word_count > 10].index
        frequent_words = [word for word in frequent_words if word in stopwords]
        alignment_freq_df = alignment_df[alignment_df.src.isin(frequent_words)]

        trg_stopwords = []

        for src_word, group in tqdm(alignment_freq_df.groupby('src'), total=len(alignment_freq_df.src.unique()), desc='Finding aligned stop words'):
            targets = group.trg
            target_counts = targets.value_counts(normalize=True).reset_index()
            target_counts.columns = ['trg', 'count']
            top_match = target_counts.iloc[0]
            if top_match['count'] < 0.2:
                continue



            target_list = list(set(targets))
            target = top_match.trg
            if primitives_to_character:
                trg_stopwords.append(target)
                continue

            matches = np.vectorize(fuzz.ratio)([target], target_list)
            idxs = np.where(matches >= 80)[0]
            words = [target_list[idx] for idx in idxs]
            # count_diff = target_counts.iloc[0]['count'] - target_counts.iloc[1]['count']
            # print(f"{src_word} -> {target} -> {words} ({len(targets)}), diff: {count_diff}")
            trg_stopwords.extend(words)

        pruned_trg_stopwords = []
        for trg in trg_stopwords:
            subset = alignment_df.query(f'trg == "{trg}"')
            if len(subset) < 10:
                # Benefit of the doubt
                pruned_trg_stopwords.append(trg)
                continue
            counts = subset.stopword.value_counts().to_dict()
            percentage = counts.get(True, 0) / len(subset)
            if percentage >= 0.3:
                pruned_trg_stopwords.append(trg)
            #else:
            #    print(f'{trg} is not a stopword: {percentage:.2f}')
        return pruned_trg_stopwords



def filter(language_tag):
    """
        Process the json dumps to create the following files:
        - `{language_tag}-filtered.csv` is a table with `word` and `count` of all words that pass the filter,
        - `{language_tag}-clean.txt` is txt file with all words that are cleaned -> used for training the compound word splitter,
        - `{language_tag}-all.txt` is txt file with all words occurring in the corpus -> used to reject pseudowords which are already in the corpus
        :param language_iso:
        :return:
        """
    all_json_files = glob(f'{data_dir}/{language_tag}/*.json')
    print(len(all_json_files), 'files')

    all_verses = {}
    for json_file in all_json_files:
        with open(json_file) as f:
            all_verses = {
                **all_verses,
                **{
                    verse_id: verse
                    for verse_id, verse in json.load(f).items()
                    # block text spanning multiple verses
                    if '+' not in verse_id
                }
            }

    print(len(all_verses), 'verses')
    all_words = [word for verse in all_verses.values() for word in verse.split(' ')]
    print(len(all_words), 'total unique words')

    os.makedirs(f'{database_dir}/{language_tag}', exist_ok=True)
    with open(f'{database_dir}/{language_tag}/{language_tag}-all.txt', 'w') as f:
        f.write('\n'.join(set(all_words)))

    word_lengths = [len(word) for word in all_words]
    median_word_length = np.median(word_lengths)
    do_uromanize = median_word_length <= 3
    info(f'Median word length: {median_word_length:.2f}, do uromanize: {do_uromanize}')
    language_iso = iso3_to_iso2.get(language_tag, None)
    cleaned_words = [word for word in all_words if basic_checks(word)[0] is False]
    print_summary('Remaining after basic checks', len(cleaned_words), len(all_words))
    if do_uromanize and language_tag not in ['kor', 'zho_tw', 'lzh', 'jpn', 'hak']:
        print("Warning: language seems to contains too short words")
    token_df = pd.Series(cleaned_words).value_counts().reset_index()
    token_df.columns = ['token', 'token_count']
    token_df, primitives_to_character = expand_word(token_df, language_iso, do_uromanize)
    if do_uromanize:
        primitives_df = pd.DataFrame({
            'primitive': list(primitives_to_character.keys()),
            'character': list(primitives_to_character.values())
        })
        primitives_df = primitives_df.dropna()
        primitives_path = f'{database_dir}/{language_tag}/{language_tag}-primitives.csv'
        primitives_df.to_csv(primitives_path, index=False)

    # Align
    if language_tag != 'eng':
        stop_words = find_aligned_stop_words(all_verses, primitives_to_character)
    else:
        stop_words = []


    old_count = token_df.shape[0]
    token_df = token_df[~token_df.token.isin(stop_words)]

    print_summary(f'Remaining after stop words', token_df.shape[0], old_count)


    n_chars, char_df, unicode_block_table = get_char_df(token_df)
    unicode_block_table['percentage'] = unicode_block_table['count'] / unicode_block_table['count'].sum()
    print(unicode_block_table.head())
    if language_iso in lang_2_script:
        accepted_scripts = lang_2_script[language_iso]
    else:
        accepted_scripts = []
        cumulative_percentage = 0
        for _, row in unicode_block_table.iterrows():
            cumulative_percentage += row['percentage']
            accepted_scripts.append(row['block'])
            if cumulative_percentage > 0.5:
                break
    allowed_words = [
        word
        for word in token_df.token.unique()
        if in_accepted_script(word, accepted_scripts, char_df)
    ]
    old_count = token_df.shape[0]
    token_df = token_df[token_df.token.isin(allowed_words)]
    print_summary(f'Remaining after checking script scripts', token_df.shape[0], old_count)

    if do_uromanize:
        info("Skipping compound word splitting")
    else:
        word_tuples = [tuple(l) for l in token_df[['token', 'token_count']].values]
        train_splitter(word_tuples, language_tag, DATASET)

        splitter = Splitter(language_tag, DATASET)

        is_compound = []
        unique_words = list(set(allowed_words))
        for word in tqdm(unique_words):
            if type(word) != str:
                continue
            is_compound.append(is_compound_word(
                word=word,
                splitter=splitter,
                word_list=unique_words,
            ))

        #compound_tokens = [unique_words[i] for i, c in enumerate(is_compound) if c]
        no_compound_tokens = [unique_words[i] for i, c in enumerate(is_compound) if not c]

        old_count = token_df.shape[0]
        token_df = token_df[token_df.token.isin(no_compound_tokens)]

        print_summary(f'Remaining after removing compound words', token_df.shape[0], old_count)


    # Only keep words with typical length
    median_length = token_df.token_expanded_length.median()
    std_length = token_df.token_expanded_length.std()
    token_length_range = [
        int(max(1, np.floor(median_length - 2 * std_length))),
        int(np.ceil(median_length + 2 * std_length))
    ]

    old_length = token_df.shape[0]

    token_df.query(
        'token_expanded_length >= @token_length_range[0] and token_expanded_length <= @token_length_range[1]',
        inplace=True)

    print_summary(f'Remaining words after removing words with atypical length', token_df.shape[0], old_length)
    token_df['count'] = token_df['token_count']
    token_df['word'] = token_df['token']
    token_df.to_csv(f'{database_dir}/{language_tag}/{language_tag}-filtered.csv', index=False)


    with open(f'{database_dir}/{language_tag}/{language_tag}-clean.txt', 'w') as f:
        f.write('\n'.join(token_df.word))





if __name__ == '__main__':
    csv = f'{data_dir}/available_bibles.csv'
    all_bibles = pd.read_csv(csv)

    parser = argparse.ArgumentParser(description='Filter bible data')
    parser.add_argument('--language', type=str, required=True, help='Tag of language',
                        choices=all_bibles['language_tag'].tolist())
    args = parser.parse_args()
    filter(args.language)

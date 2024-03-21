import pandas as pd
from glob import glob

import argparse
import os
import sys
import random

import json
import re
from tqdm import tqdm
import requests
import time
import urllib.request

from vocabtest.utils.checks import basic_checks

fasttext_fallback = True


# Workaround for ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1129)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


start_time = time.time()

MODEL_MAPPING = {
    'af': 'afrikaans-afribooms-ud-2.10-220711',
    'ar': 'arabic-padt-ud-2.10-220711',
    'be': 'belarusian-hse-ud-2.10-220711',
    'bg': 'bulgarian-btb-ud-2.10-220711',
    'ca': 'catalan-ancora-ud-2.10-220711',
    'cs': 'czech-cac-ud-2.10-220711',
    'cy': 'welsh-ccg-ud-2.10-220711',
    'da': 'danish-ddt-ud-2.10-220711',
    'de': 'german-hdt-ud-2.10-220711',
    'el': 'greek-gdt-ud-2.10-220711',
    'en': 'english-partut-ud-2.10-220711',
    'es': 'spanish-ancora-ud-2.10-220711',
    'et': 'estonian-edt-ud-2.10-220711',
    'eu': 'basque-bdt-ud-2.10-220711',
    'fa': 'persian-perdt-ud-2.10-220711',
    'fi': 'finnish-tdt-ud-2.10-220711',
    'fo': 'faroese-farpahc-ud-2.10-220711',
    'fr': 'french-partut-ud-2.10-220711',
    'ga': 'irish-idt-ud-2.10-220711',
    'gd': 'scottish_gaelic-arcosg-ud-2.10-220711',
    'gl': 'galician-ctg-ud-2.10-220711',
    'got': 'gothic-proiel-ud-2.10-220711',
    'he': 'hebrew-htb-ud-2.10-220711',
    'hi': 'hindi-hdtb-ud-2.10-220711',
    'hr': 'croatian-set-ud-2.10-220711',
    'hu': 'hungarian-szeged-ud-2.10-220711',
    'hy': 'armenian-bsut-ud-2.10-220711',
    'hyw': 'western_armenian-armtdp-ud-2.10-220711',
    'id': 'indonesian-gsd-ud-2.10-220711',
    'is': 'icelandic-modern-ud-2.10-220711',
    'it': 'italian-partut-ud-2.10-220711',
    'ja': 'japanese-gsd-ud-2.10-220711',
    'ko': 'korean-kaist-ud-2.10-220711',
    'la': 'latin-llct-ud-2.10-220711',
    'lt': 'lithuanian-hse-ud-2.10-220711',
    'lv': 'latvian-lvtb-ud-2.10-220711',
    'mt': 'maltese-mudt-ud-2.10-220711',
    'mr': 'marathi-ufal-ud-2.10-220711',
    'nl': 'dutch-alpino-ud-2.10-220711',
    'nn': 'norwegian-nynorsklia-ud-2.10-220711',
    'no': 'norwegian-bokmaal-ud-2.10-220711',
    'pl': 'polish-lfg-ud-2.10-220711',
    'pt': 'portuguese-gsd-ud-2.10-220711',
    'ro': 'romanian-simonero-ud-2.10-220711',
    'ru': 'russian-syntagrus-ud-2.10-220711',
    'sa': 'sa_vedic-ud-2.10-220711',
    'se': 'sme_giella-ud-2.10-220711',
    'sk': 'slovak-snk-ud-2.10-220711',
    'sl': 'slovenian-ssj-ud-2.10-220711',
    'sr': 'serbian-set-ud-2.10-220711',
    'sv': 'swedish-talbanken-ud-2.10-220711',
    'ta': 'tamil-ttb-ud-2.10-220711',
    'te': 'telugu-mtg-ud-2.10-220711',
    'tr': 'turkish-tourism-ud-2.10-220711',
    'ug': 'uyghur-udt-ud-2.10-220711',
    'uk': 'ukrainian-iu-ud-2.10-220711',
    'ur': 'urdu-udtb-ud-2.10-220711',
    'vi': 'vietnamese-vtb-ud-2.10-220711',
    'wo': 'wolof-wtb-ud-2.10-220711',
    'zh': 'chinese-gsdsimp-ud-2.10-220711'
}



ISO_CODES = list(MODEL_MAPPING.keys())
BLOCK_TYPES = ['Foreign', 'Abbr', 'Typo']

TEXT_BLOCKS = ['_START_ARTICLE_', '_START_PARAGRAPH_', '_START_SECTION_', '_END_SECTION_', '_END_PARAGRAPH_', '_END_ARTICLE_']

def silent_guess_language(word):
    from guess_language import guess_language
    save_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    out = guess_language(word)
    sys.stdout = save_stdout
    return out

def spellcheck(word):
    global word2spellcheck, fasttext_model, fasttext_fallback
    if word not in word2spellcheck:
        embedding = fasttext_model.predict(word, k=10)
        langs = [lang.replace("__label__", "") for lang in embedding[0]]
        fasttext_prediction = langs[0]
        fasttext_predictions = dict(zip(langs, embedding[1]))
        dictionary_prediction = silent_guess_language(word) if not fasttext_fallback else None
        word2spellcheck[word] = dictionary_prediction, fasttext_prediction, fasttext_predictions
    return word2spellcheck[word]



def reject_word(word, token, not_in_locale):
    if token['feats'] is not None:
        for key in BLOCK_TYPES:
            if key in token['feats'] and token['feats'][key] == 'Yes':
                return True, key

    is_rejected, reason = basic_checks(word)
    if is_rejected:
        return True, reason

    if not word.isalnum():
        return True, 'contains_non_alphanumeric'

    if not_in_locale:
        return True, 'wrong_locale'

    return False, None


def get_vocab(language_iso, text, model, port=None, api_type=None):
    global fasttext_fallback
    import conllu
    vocab = {}
    clean_text = text.replace('\n', '')
    params = {
        'tokenizer': '',
        'tagger': '',
        'model': model,
        'data': clean_text
    }
    if api_type is None:
        if port is not None:
            api_type = 'local'
        else:
            api_type = 'remote'
    if api_type == 'local':
        endpoint = f'http://localhost:{port}/process'
    else:
        endpoint = 'https://lindat.mff.cuni.cz/services/udpipe/api/process'
    print(f'Sending request to UDPipe at {endpoint}...')
    response = requests.post(endpoint, data=params)

    if response.status_code != 200:
        raise Exception(
            f'UDPipe request failed with status code {response.status_code} for hashed text {hash(text)} response {response.text}')
    else:
        sentences = conllu.parse(json.loads(response.text)['result'])
        for sentence in sentences:
            for token in sentence:
                lemma = token['lemma']
                if lemma == '_':
                    lemma = token['form']
                form = token['form']
                POS = token['upostag']
                lemma_locale_dict, lemma_locale_ft, lemma_predictions = spellcheck(lemma)
                lemma_locale = lemma_locale_ft if fasttext_fallback else lemma_locale_dict
                lemma_rejected, lemma_reason = reject_word(
                    lemma,
                    token,
                    lemma_locale != language_iso
                )
                if lemma not in vocab:
                    lemma_dict = {
                        'tokens': {},
                        'count': 0,
                        'locale_dictionary': lemma_locale_dict,
                        'locale_fasttext': lemma_locale_ft,
                        'locale_predictions': json.dumps(lemma_predictions),
                        'rejected': lemma_rejected,
                        'reason': lemma_reason
                    }
                else:
                    lemma_dict = vocab[lemma]

                token_locale_dict, token_locale_ft, token_predictions = spellcheck(form)
                token_locale = token_locale_ft if fasttext_fallback else token_locale_dict
                token_rejected, token_reason = reject_word(
                    form,
                    token,
                    token_locale
                )

                if form not in lemma_dict['tokens']:
                    lemma_dict['tokens'][form] = {
                        'count': 1,
                        'locale_dictionary': token_locale_dict,
                        'locale_fasttext': token_locale_ft,
                        'locale_predictions': json.dumps(token_predictions),
                        'POS': POS,
                        'rejected': token_rejected,
                        'reason': token_reason
                    }
                else:
                    lemma_dict['tokens'][form]['count'] += 1

                lemma_dict['count'] += 1
                vocab[lemma] = lemma_dict
    return vocab


def get_time(msg):
    global start_time
    print(msg + " in %s seconds" % (time.time() - start_time))
    start_time = time.time()
def process_loop(language_iso, reversed_order=False, port=None):
    global fasttext_fallback
    import enchant
    import fasttext
    broker = enchant.Broker()
    lang_list = list(set([lang.split('_')[0] for lang in broker.list_languages()]))
    fasttext_fallback = not language_iso in lang_list
    print("Using dictionary" if not fasttext_fallback else "Using fasttext")

    assert language_iso in MODEL_MAPPING.keys(), f'Language {language_iso} not supported'
    model = MODEL_MAPPING[language_iso]
    print(model)

    global word2spellcheck, fasttext_model
    word2spellcheck = {}
    fasttext_checkpoint = f'{package_dir}/wikipedia/dependencies/lid.176.bin'
    os.makedirs('dependencies', exist_ok=True)
    if not os.path.exists(fasttext_checkpoint):
        print('Downloading fasttext model...')
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', fasttext_checkpoint)
    fasttext_model = fasttext.load_model(fasttext_checkpoint)

    json_path = f'{package_dir}/wikipedia/data/wikipedia_{language_iso}_10000_longest_articles.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    if reversed_order:
        print('Reversing order of articles...')
        print('Limiting to first 10,000...')
        data = data[:10000]
        data = data[::-1]
    n_articles = len(data)

    os.makedirs(f'{package_dir}/wikipedia/json/{language_iso}', exist_ok=True)

    # Set up the connection
    for idx, row in tqdm(enumerate(data), desc=f'Processing {n_articles} wiki articles for {language_iso}'):
        if 'id' in row:
            wiki_id = row['id']
        else:
            wiki_id = row['wikidata_id']

        if 'url' in row:
            url = row['url']
        else:
            url = row['version_id']

        dump_file = f'{package_dir}/wikipedia/json/{language_iso}/vocab_{language_iso}_{wiki_id}.json'
        if os.path.exists(dump_file):
            continue

        # Drop footnote superscripts in brackets
        text = row['text']
        text = re.sub(r"\[.*?\]+", '', text)
        text = re.sub(r"\(.*?\)+", '', text)
        text = re.sub(r"\{.*?\}+", '', text)
        text = text.replace('_NEWLINE_', '\n')  # _NEWLINE_
        for remove_str in TEXT_BLOCKS:
            text = text.replace(remove_str, '')
        text = text.replace('\n', ' ')

        tries = 0
        while True:
            try:
                tries += 1
                if tries > 10:
                    break
                vocab = get_vocab(language_iso, text, model, port)
                with open(dump_file, 'w') as f:
                    json.dump(vocab, f)
                break
            except Exception as e:
                time.sleep(random.randint(1, 10))
                print(f"Error processing {wiki_id} {url} {e}")

    print(f'Processed articles: {n_articles}')

def merge(language_iso):
    relevant_columns = ['locale_dictionary', 'locale_fasttext', 'POS', 'rejected', 'reason']
    os.makedirs(f'{package_dir}/wikipedia/databases/{language_iso}', exist_ok=True)
    token_dump_path = f'{package_dir}/wikipedia/databases/{language_iso}/{language_iso}.csv'

    vocab = {}
    results = []
    for path in tqdm(glob(f'{package_dir}/wikipedia/json/{language_iso}/vocab_{language_iso}_*.json')):
        new_vocab = json.load(open(path, 'r'))
        for lemma, lemma_dict in new_vocab.items():
            for token, token_dict in lemma_dict['tokens'].items():
                results.append({
                    'lemma': lemma,
                    'token': token,
                    'token_count': token_dict['count'],
                    'article': path.split('_')[-1].split('.')[0],
                    **{col: token_dict[col] for col in relevant_columns}
                })
            if lemma not in vocab:
                vocab[lemma] = lemma_dict
            else:
                vocab[lemma]['count'] += lemma_dict['count']
                for token_key in lemma_dict['tokens'].keys():
                    if token_key in vocab[lemma]['tokens']:
                        vocab[lemma]['tokens'][token_key]['count'] += 1
                    else:
                        vocab[lemma]['tokens'][token_key] = lemma_dict['tokens'][token_key]

    results_df = pd.DataFrame(results)
    results_df.to_csv(token_dump_path, index=False)


import argparse
from os.path import exists
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm
import json

from vocabtest import package_dir

# Workaround for ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1129)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def download_wiki40b(language_iso):
    return load_dataset("wiki40b", language_iso, beam_runner='DirectRunner')


def download_wikipedia(language_iso):
    source = urlopen(f'https://dumps.wikimedia.org/{language_iso}wiki/').read()
    soup = BeautifulSoup(source, 'html.parser')
    latest_dump = sorted([a['href'] for a in soup.find_all('a')])[-2].replace('/', '')
    print('Downloading latest dump:', latest_dump)
    return load_dataset("wikipedia", language=language_iso, date=latest_dump, beam_runner='DirectRunner')


def _download(language_iso, top_n=10000):
    json_file = f'{package_dir}/wikipedia/data/wikipedia_{language_iso}_{top_n}_longest_articles.json'
    if exists(json_file):
        print(f'File {json_file} already exists, skipping')
        return json.load(open(json_file))

    if language_iso in ['nn', 'hy', 'gl']:
        wiki_data = download_wikipedia(language_iso)
    else:
        try:
            wiki_data = download_wiki40b(language_iso)
        except:
            print(f"No preprocessed dump available for {language_iso}")
            wiki_data = download_wikipedia(language_iso)

    n_chars_df = pd.DataFrame({})
    for split in ['test', 'train', 'validation']:
        if split in wiki_data:
            n_chars = []
            for article in tqdm(wiki_data[split]):
                n_chars.append(len(article['text']))

            n_chars_df = pd.concat([n_chars_df, pd.DataFrame({
                'n_chars': n_chars,
                'idx': range(len(n_chars)),
                'split': split
            })])
    longest_articles = n_chars_df.sort_values('n_chars', ascending=False).head(top_n)
    data = []
    for _, row in tqdm(longest_articles.iterrows()):
        idx = row['idx']
        split = row['split']
        article = wiki_data[split][idx]
        article['text'] = article['text'].replace('_NEWLINE_', '\n')  # _NEWLINE_
        for remove_str in ['_START_ARTICLE_', '_START_PARAGRAPH_', '_START_SECTION_', '_END_SECTION_',
                           '_END_PARAGRAPH_', '_END_ARTICLE_']:
            article['text'] = article['text'].replace(remove_str, '')
        data.append(article)

    with open(json_file, 'w') as f:
        json.dump(data, f)

    return data

def download(language_iso, reversed_order=False, port=None, top_n=10000):
    print(f'Downloading {language_iso}...')
    _download(language_iso, top_n)
    print(f'Processing {language_iso}...')
    process_loop(language_iso, reversed_order, port)
    print(f'Merging {language_iso}...')
    merge(language_iso)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', required=True, help='ISO language_iso code')
    parser.add_argument('--reversed_order', default=0, type=int, help='Go through the data in reversed order', choices=[0, 1])
    parser.add_argument('--port', default=None, help='Port for the server')
    parser.add_argument('--top_n', type=int, default=10000)
    args = parser.parse_args()
    args.reversed_order = bool(args.reversed_order)
    download(args.language, args.reversed_order, args.port, args.top_n)

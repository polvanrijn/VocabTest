import argparse
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
import random

import apsw
import pandas as pd
import json
import conllu
import re
from tqdm import tqdm
import requests
import time
import fasttext
from guess_language import guess_language
from os.path import exists

start_time = time.time()


def get_time(msg):
    global start_time
    print(msg + " in %s seconds" % (time.time() - start_time))
    start_time = time.time()


parser = argparse.ArgumentParser()
parser.add_argument('--language', required=True, help='ISO language code')
parser.add_argument('--reversed_order', default=0, type=int, help='Go through the data in reversed order', choices=[0, 1])
parser.add_argument('--port', default=None, help='Port for the server')
args = parser.parse_args()
args.reversed_order = bool(args.reversed_order)

import enchant
broker = enchant.Broker()
lang_list = list(set([lang.split('_')[0] for lang in broker.list_languages()]))
fasttext_fallback = not args.language in lang_list
print("Using dictionary" if not fasttext_fallback else "Using fasttext")

domain = f"https://{args.language}.wikipedia.org"
model_mapping = {
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

ud_models = pd.read_csv('ud2-10_models.csv', names=['names', 'path', 'variant', 'acknowledgements'])

assert args.language in model_mapping.keys(), f'Language {args.language} not supported'
MODEL = model_mapping[args.language]
print(MODEL)

BLOCK_TYPES = ['Foreign', 'Abbr', 'Typo']
PUNCTUATION = ['|', '\\', '^', ',', '(', '*', '"', '!', '$', '/', '[', '`', ';', ']', '#', '}', '&', '=', "'", '@', '~',
               '{', '>', '<', '%', '_', '?', '+', '-', ')', ':', '.']
TEXT_BLOCKS = ['_START_ARTICLE_', '_START_PARAGRAPH_', '_START_SECTION_', '_END_SECTION_', '_END_PARAGRAPH_', '_END_ARTICLE_']
word2spellcheck = {}
fasttext_model = fasttext.load_model("fastText/lid.176.bin")

def silent_guess_language(word):
    save_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    out = guess_language(word)
    sys.stdout = save_stdout
    return out

def spellcheck(word):
    global word2spellcheck
    if word not in word2spellcheck:
        embedding = fasttext_model.predict(word, k=10)
        langs = [lang.replace("__label__", "") for lang in embedding[0]]
        fasttext_prediction = langs[0]
        fasttext_predictions = dict(zip(langs, embedding[1]))
        dictionary_prediction = silent_guess_language(word) if not fasttext_fallback else None
        word2spellcheck[word] = dictionary_prediction, fasttext_prediction, fasttext_predictions
    return word2spellcheck[word]


def reject_word(word, token, locale):
    if token['feats'] is not None:
        for key in BLOCK_TYPES:
            if key in token['feats'] and token['feats'][key] == 'Yes':
                return True, key
    
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
    
    if not word.isalnum():
        return True, 'contains_non_alphanumeric'
    
    if args.language != locale:
        return True, 'wrong_locale'

    return False, None


def get_vocab(text, API = None):
    vocab = {}
    clean_text = text.replace('\n', '')
    params = {
        'tokenizer': '',
        'tagger': '',
        'model': MODEL,
        'data': clean_text
    }
    if API is None:
        if args.port is not None:
            API = 'local'
        else:
            API = 'remote'
    if API == 'local':
        endpoint = f'http://localhost:{args.port}/process'
    else:
        endpoint = 'https://lindat.mff.cuni.cz/services/udpipe/api/process'
    print(f'Sending request to UDPipe at {endpoint}...')
    response = requests.post(endpoint, data=params)

    if response.status_code != 200:
        raise Exception(f'UDPipe request failed with status code {response.status_code} for hashed text {hash(text)} response {response.text}')
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
                lemma_rejected, lemma_reason = reject_word(
                    lemma, 
                    token, 
                    lemma_locale_ft if fasttext_fallback else lemma_locale_dict
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
                token_rejected, token_reason = reject_word(
                    form,
                    token,
                    token_locale_ft if fasttext_fallback else token_locale_dict
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

if args.language in ["got", "wo", "se", "gd", "mt", "fo", "lv", "et", "lt", "ga", "cy", "is", "mr", "la", "sk", "sa", "hyw", "be"]:
    json_path = f'data/wikipedia_{args.language}.json'
else:
    json_path = f'data/wikipedia_{args.language}_10000_longest_articles.json'

with open(json_path, 'r') as f:
    data = json.load(f)

if args.reversed_order:
    print('Reversing order of articles...')
    print('Limiting to first 10,000...')
    data = data[:10000]
    data = data[::-1]
n_articles = len(data)
vocab = {}

os.makedirs(f'json/{args.language}', exist_ok=True)

# Setup the connection
for idx, row in tqdm(enumerate(data), desc=f'Processing {n_articles} wiki articles for {args.language}'):
    if 'id' in row:
        wiki_id = row['id']
    else:
        wiki_id = row['wikidata_id']

    if 'url' in row:
        url = row['url']
    else:
        url = row['version_id']

    dump_file = f'json/{args.language}/vocab_{args.language}_{wiki_id}.json'
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
            vocab = get_vocab(text)
            with open(dump_file, 'w') as f:
                json.dump(vocab, f)
            break
        except:
            time.sleep(random.randint(1, 10))
            print(f"Error processing {wiki_id} {url}")

print(f'Processed articles: {n_articles}')

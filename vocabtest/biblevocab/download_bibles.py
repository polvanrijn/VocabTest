from glob import glob
from os import makedirs
import re
from os.path import exists
from time import sleep

from icu_tokenizer import SentSplitter, Normalizer, Tokenizer
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import json

from vocabtest import biblevocab_dir

import argparse
args = argparse.ArgumentParser()
args.add_argument('--language', type=str, required=True)
args = args.parse_args()

with open(f'{biblevocab_dir}/available_bibles.json', 'r') as f:
    data = json.load(f)


language_iso = args.language

splitter = SentSplitter(language_iso)
normalizer = Normalizer(language_iso, norm_puncts=True)
tokenizer = Tokenizer(language_iso)

for row in data:
    if row['iso'] == language_iso:
        break

assert row['iso'] == language_iso

PUNCTUATION = ['.', '! ', '? ', ';', ]

def request(url, max_retries=10):
    for i in range(max_retries):
        try:
            return requests.get(url)
        except:
            print(f'Failed to get {url} (retry {i}, trying again in {2**i} seconds)')
            sleep(2**i)
            continue
    raise Exception(f'Failed to get {url}')

for version in row['versions']:
    version_id = version['id']
    version_abr = version['local_abbreviation']
    for book in version['books']:
        book_id = book['usfm']
        try:
            chapters = request(f'https://www.bible.com/json/bible/books/{version_id}/{book_id}/chapters').json()['items']
        except:
            print(f'Failed to get chapters for {version_abr} {book_id}')
            continue

        dir = f'data/{language_iso}/{version_abr}/{book_id}/'
        makedirs(dir, exist_ok=True)
        if len(glob(f'{dir}/*.json')) == len(chapters):
            print(f'Already downloaded {version_abr} {book_id}')
            continue

        for chapter in tqdm(chapters, desc=f'Downloading chapters {version_abr} {book_id}'):
            chapter_id = chapter["usfm"]
            json_file = f'{dir}/{chapter_id}.json'
            if exists(json_file):
                continue
            try:
                response = request(f'https://www.bible.com/bible/{version_id}/{chapter_id}')
            except:
                print(f'Failed to get {chapter_id}')
                continue
            soup = BeautifulSoup(response.content, 'html.parser')
            try:
                verses = soup.find('div', class_='book').find_all('span', class_='verse')
            except:
                continue
            verse_dict = {}
            for verse in verses:
                verse_id = verse["data-usfm"]
                if verse.text == ' ':
                    continue
                try:
                    text = verse.find('span', class_='content').text.strip()
                    text = text.lower()
                    text = re.sub(r'[^\w\s]', '', text) # Strip punctuation
                    tokens = tokenizer.tokenize(normalizer.normalize(text))
                    text = ' '.join(tokens)
                    if text == '':
                        continue
                    verse_dict[verse_id] = text
                except:
                    continue

            if len(verse_dict) == 0:
                continue
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(verse_dict, f, ensure_ascii=False, indent=4)


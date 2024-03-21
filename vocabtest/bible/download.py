import os

import argparse

import pandas as pd
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

from vocabtest.utils.iso import iso3_to_iso2
from vocabtest import package_dir

DATASET = 'bible'
data_dir = f'{package_dir}/{DATASET}/data'
os.makedirs(data_dir, exist_ok=True)

csv = f'{data_dir}/available_bibles.csv'
if os.path.exists(csv):
    all_bibles = pd.read_csv(csv)
    all_bibles.index = all_bibles.language_tag

else:
    all_bibles = pd.DataFrame(
        requests.get('https://www.bible.com/api/bible/configuration').json()["response"]["data"]["default_versions"]
    )
    all_bibles.index = all_bibles.language_tag
    all_bibles.to_csv(csv)


def request(url, max_retries=10):
    for i in range(max_retries):
        try:
            return requests.get(url)
        except:
            print(f'Failed to get {url} (retry {i}, trying again in {2 ** i} seconds)')
            sleep(2 ** i)
            continue
    raise Exception(f'Failed to get {url}')


# Structure:
# - language
#  - version
#   - book
#    - chapter
#     - verse

def download(language_tag):
    language_iso = iso3_to_iso2.get(language_tag, language_tag)
    if language_iso == language_tag:
        print(f'No ISO2 code for {language_iso}')

    normalizer = Normalizer(language_iso, norm_puncts=True)
    tokenizer = Tokenizer(language_iso)

    language = all_bibles.loc[language_tag]
    versions = request(f'https://www.bible.com/json/bible/versions/{language_tag}').json()['items']
    if len(versions) == 0:
        print(f'No versions for {language_tag}')
        return
    error = False
    try:
        version_id = int(language['id'])
        version_name = [version['local_abbreviation'] for version in versions if version['id'] == version_id][0]
    except:
        version_name = versions[0]['local_abbreviation']
        version_id = int(versions[0]['id'])
    books = request(f'https://www.bible.com/json/bible/books/{version_id}').json()['items']
    for book in tqdm(books, desc=f'Downloading books {version_name}', total=len(books)):
        book_id = book['usfm']
        try:
            chapters = request(f'https://www.bible.com/json/bible/books/{version_id}/{book_id}/chapters').json()[
                'items']
        except:
            print(f'Failed to get chapters for {version_name} {book_id}')
            continue

        dir = f'{data_dir}/{language_tag}/'
        makedirs(dir, exist_ok=True)
        if len(glob(f'{dir}/*.json')) == len(chapters):
            print(f'Already downloaded {version_name} {book_id}')
            continue

        for chapter in chapters:
            chapter_id = chapter["usfm"]
            if 'INTRO' in chapter_id:
                continue
            json_file = f'{dir}/{chapter_id}.json'
            if exists(json_file):
                continue

            try_again = True
            verses = []
            try:
                response = request(
                    f'https://www.bible.com/_next/data/tXhcyy6O6HyMaq7vF4HYA/en/bible/{version_id}/{chapter_id}.{version_name}.json')
                html = response.json()['pageProps']['chapterInfo']['content']
                soup = BeautifulSoup(html, 'html.parser')
                verses = soup.find_all('span', class_='verse')
                verses = [
                    {
                        'data-usfm': verse["data-usfm"],
                        'text': verse.find('span', class_='content').text,
                    }
                    for verse in verses
                ]
                try_again = False
            except:
                pass

            if try_again:
                try:
                    response = request(f'https://www.bible.com/bible/{version_id}/{chapter_id}.{language_tag.upper()}')
                    html = response.content
                    soup = BeautifulSoup(html, 'html.parser')
                    verses = [
                        {
                            'data-usfm': span.attrs['data-usfm'],
                            'text': span.find_all('span')[-1].text,
                        }
                        for span in soup.find_all('span')
                        if 'data-usfm' in span.attrs
                    ]
                except Exception as e:
                    print(f'Failed to get {chapter_id} {version_name} {e}')
                    continue
            verse_dict = {}
            for verse in verses:
                verse_id = verse["data-usfm"]
                if verse['text'] == ' ':
                    continue
                try:
                    text = verse['text'].strip()
                    text = text.lower()
                    text = re.sub(r'[^\w\s]', '', text)  # Strip punctuation
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download bible data')
    parser.add_argument('--language', required=True, type=str, help='Tag of language to download',
                        choices=all_bibles['language_tag'].tolist())
    args = parser.parse_args()
    download(args.language)

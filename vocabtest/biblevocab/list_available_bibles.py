"""
Lists all available bibles on bible.com
"""

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json

from vocabtest import biblevocab_dir

response = requests.get('https://www.bible.com/versions')
soup = BeautifulSoup(response.content, 'html.parser')
container = soup.find('div', class_='pr4-m')

def save_json(all_languages):
    with open(f'{biblevocab_dir}/available_bibles.json', 'w') as f:
        json.dump(all_languages, f, indent=4)

all_languages = []
for i, elem in tqdm(enumerate(container), desc='Scraping languages'):
    if i % 10 == 0:
        save_json(all_languages)
    try:
        if elem.name == 'a' and elem['href'].startswith('/languages'):
            iso = elem['href'].replace('/languages/', '').split('-')[0]
            if iso == "mis":
                continue
            title = elem.text
            versions = requests.get(f'https://www.bible.com/json/bible/versions/{iso}').json()['items']
            for version in tqdm(versions, desc=f"Scraping versions for {title}"):
                version_id = version['id']
                books = requests.get(f'https://www.bible.com/json/bible/books/{version_id}').json()['items']
                # for book in books:
                #     book_id = book['usfm']
                #     chapters = requests.get(f'https://www.bible.com/json/bible/books/{version_id}/{book_id}/chapters').json()['items']
                #     for chapter in tqdm(chapters):
                #         chapter_id = chapter['usfm']
                #         response = requests.get(f'https://www.bible.com/bible/{version_id}/{chapter_id}')
                #         soup = BeautifulSoup(response.content, 'html.parser')
                #
                #         verses = soup.find('div', class_='book').find_all('span', class_='verse')
                #         verse_dict = {}
                #         for verse in verses:
                #             verse_id = verse["data-usfm"]
                #             if verse.text == ' ':
                #                 continue
                #             text = verse.find('span', class_='content').text.strip()
                #             verse_dict[verse_id] = text
                #         chapter['verses'] = verse_dict
                #     book['chapters'] = chapters
                version['books'] = books
            all_languages.append({
                'iso': iso,
                'title': title,
                'versions': versions
            })
    except Exception as e:
        print(e)

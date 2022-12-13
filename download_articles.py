import argparse
from os.path import exists
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm
import json


parser = argparse.ArgumentParser()
parser.add_argument('--language', required=True, help='ISO language code')
parser.add_argument('--top_n', default=10000, type=int, help='N longest articles in that language')
args = parser.parse_args()

json_file = f'data/wikipedia_{args.language}_{args.top_n}_longest_articles.json'
if exists(json_file):
    print(f'File {json_file} already exists, skipping')
    exit()

def download_wiki40b():
    return load_dataset("wiki40b", args.language, beam_runner='DirectRunner')

def download_wikipedia():
    source = urlopen(f'https://dumps.wikimedia.org/{args.language}wiki/').read()
    soup = BeautifulSoup(source, 'html.parser')
    latest_dump = sorted([a['href'] for a in soup.find_all('a')])[-2].replace('/', '')
    print('Downloading latest dump:', latest_dump)
    return load_dataset("wikipedia", language=args.language, date=latest_dump, beam_runner='DirectRunner')

if args.language in ['nn', 'hy', 'gl']:
    wiki_data = download_wikipedia()
else:
    try:
        wiki_data = download_wiki40b()
    except:
        print(f"No preprocessed dump available for {args.language}")
        wiki_data = download_wikipedia()

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
longest_articles = n_chars_df.sort_values('n_chars', ascending=False).head(args.top_n)
data = []
for _, row in tqdm(longest_articles.iterrows()):
    idx = row['idx']
    split = row['split']
    article = wiki_data[split][idx]
    article['text'] = article['text'].replace('_NEWLINE_', '\n')  # _NEWLINE_
    for remove_str in ['_START_ARTICLE_', '_START_PARAGRAPH_', '_START_SECTION_', '_END_SECTION_', '_END_PARAGRAPH_', '_END_ARTICLE_']:
        article['text'] = article['text'].replace(remove_str, '')
    data.append(article)

with open(json_file, 'w') as f:
    json.dump(data, f)
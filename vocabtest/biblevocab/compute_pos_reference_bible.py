import re
import pandas as pd
import json
import spacy

from glob import glob
from tqdm import tqdm
from vocabtest import dependencies_dir, biblevocab_data_dir

names = set(
    pd.read_csv(f'{dependencies_dir}/theographic-bible-metadata/CSV/Books.csv').bookName.tolist() +
    pd.read_csv(f'{dependencies_dir}/theographic-bible-metadata/CSV/People.csv').displayTitle.tolist() +
    pd.read_csv(f'{dependencies_dir}/theographic-bible-metadata/CSV/PeopleGroups.csv').groupName.tolist() +
    pd.read_csv(f'{dependencies_dir}/theographic-bible-metadata/CSV/Places.csv').displayTitle.tolist()
)
block_set = set([re.sub("[\(\[].*?[\)\]]", "", name.lower()) for name in names if type(name) == str and ' ' not in name])

nlp = spacy.load("en_core_web_sm")

bible_names = sorted(glob(f'{biblevocab_data_dir}/eng/NRSVUE/*/*.json'))

bible = {}

for bible_name in bible_names:
    with open(bible_name) as f:
        bible = {
            **bible,
            **json.load(f)
        }

with open(f'{biblevocab_data_dir}/eng/NRSVUE.json', 'w') as f:
    json.dump(bible, f)


block_idx = {}
valid_tokens = []
for verse_id, verse in tqdm(bible.items()):
    block_idx[verse_id] = []
    for idx, token in enumerate(nlp(verse)):
        if token.pos_ == 'PROPN':
            block_idx[verse_id].append(idx)
        elif token.text in block_set:
            block_idx[verse_id].append(idx)
        else:
            valid_tokens.append(token.text)

with open(f'{biblevocab_data_dir}/eng/NRSVUE_block_idx.json', 'w') as f:
    json.dump(block_idx, f)
import hashlib
import os
from glob import glob
from os.path import basename, join

import pandas as pd
from tqdm import tqdm
import argparse

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from bidi import algorithm as bidialg
import arabic_reshaper


#
# paths = glob(join(package_dir, 'pairs', 'v2', 'minified', '*.csv'))
# for path in paths:
#     if pd.read_csv(path).shape[0] != 1000:
#         print(pd.read_csv(path).shape[0])
#         print(path)
# assert len(paths) == 60
#
# all_dfs = []
#
# for path in paths:
#     df = pd.read_csv(path)
#     df['word_length'] = df.stimulus.apply(lambda x: len(x))
#     language = basename(path).replace('.csv', '')
#     df['language'] = basename(path).replace('.csv', '')
#     df['hash'] = df.stimulus.apply(lambda text: hashlib.md5(text.encode('utf-8')).hexdigest())
#
#     paths = [f'pairs/v2/images/{language}/{hash}.svg' for hash in df.hash]
#     os.path.exists(paths[0])
#
#     all_dfs.append(df)
#
# df = pd.concat(all_dfs)
#

def plot_text(text, file_path, font_name):
    plt.text(
        0.5,
        0.5,
        text,
        family=[font_name],
        fontsize=34,
        weight='black',
        horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    plt.savefig(file_path)
    plt.close()
    plt.cla()


def process_df(df, language, font_folder, output_folder):
    font_files = fm.findSystemFonts(fontpaths=font_folder)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)

    pb = tqdm(df.iterrows(), total=len(df))

    if language in ['ar', 'fa', 'ug', 'ur']:
        font_name = 'Noto Sans Arabic'
    elif language == "got":
        font_name = 'Noto Sans Gothic'
    elif language in ['he', 'yi']:
        font_name = 'Noto Sans Hebrew'
    elif language in ['hi', 'mr', 'sa']:
        font_name = 'Noto Sans Devanagari'
    elif language in ["hy", "hyw"]:
        font_name = 'Noto Sans Armenian'
    elif language == 'ja':
        font_name = 'Noto Sans JP'
    elif language == 'ko':
        font_name = 'Noto Sans KR Black'
    elif language == 'ta':
        font_name = 'Noto Sans Tamil'
    elif language == 'te':
        font_name = 'Noto Sans Telugu'
    elif language == 'zh':
        font_name = 'Noto Sans SC Black'
    else:
        font_name = 'Inter'

    paths = []
    for _, row in pb:
        text = row.stimulus

        # hash the text to get a unique filename
        hash = hashlib.md5(text.encode('utf-8')).hexdigest()

        pb.set_description(f"{language} {text}")

        img_dir = f'{output_folder}/{language}'
        os.makedirs(img_dir, exist_ok=True)
        img_name = f'{img_dir}/{hash}.svg'
        if language == 'ar':
            text = arabic_reshaper.reshape(text)
        plot_text(bidialg.get_display(text), img_name, font_name)
        paths.append(img_name)
    return paths


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--language', type=str, required=True)
    argparser.add_argument('--csv', type=str, required=True)
    argparser.add_argument('--hash_secret', type=str, default='')
    argparser.add_argument('--font_folder', type=str, default='fonts')
    argparser.add_argument('--output_folder', type=str, default='images')
    args = argparser.parse_args()

    df = pd.read_csv(args.csv)
    df.stimulus.apply(lambda text: hashlib.md5((text + args.hash_secret).encode('utf-8')).hexdigest())
    paths = process_df(df, args.language, args.font_folder, args.output_folder)

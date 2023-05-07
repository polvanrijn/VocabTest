import json
import os
import subprocess
from glob import glob
from os.path import exists
import pandas as pd

from vocabtest import biblevocab_data_dir


def load_bibles(target_bible_path, language_iso, version):
    # Load reference and target bible
    with open(f'{biblevocab_data_dir}/eng/NRSVUE.json') as f:
        reference_bible = json.load(f)

    with open(f'{biblevocab_data_dir}/eng/NRSVUE_block_idx.json') as f:
        reference_bible_block_idx = json.load(f)

    if exists(target_bible_path):
        with open(target_bible_path) as f:
            target_bible = json.load(f)
    else:
        target_bible = {}
        for bible_name in glob(f'{biblevocab_data_dir}/{language_iso}/{version}/*/*.json'):
            with open(bible_name) as f:
                target_bible = {
                    **target_bible,
                    **json.load(f)
                }
        with open(target_bible_path, 'w') as f:
            json.dump(target_bible, f)

    return reference_bible, reference_bible_block_idx, target_bible


def match_bibles(reference_bible, target_bible, giza_folder):
    # Match reference and target bible by verse id
    reference_lines = []
    target_lines = []
    verse_ids = []
    for reference_verse_id, reference_verse in reference_bible.items():
        if reference_verse_id in target_bible:
            verse_ids.append(reference_verse_id)
            target_verse = target_bible[reference_verse_id]
            reference_lines.append(reference_verse + '\n')
            target_lines.append(target_verse + '\n')
    assert len(reference_lines) == len(target_lines)

    # Dump the aligned verses to a file for GIZA++
    os.makedirs(giza_folder, exist_ok=True)
    ref_path = f'{giza_folder}/reference.txt'
    with open(ref_path, 'w') as reference_file:
        reference_file.writelines(reference_lines)

    target_path = f'{giza_folder}/target.txt'
    with open(target_path, 'w') as target_file:
        target_file.writelines(target_lines)

    return verse_ids, ref_path, target_path

def align_with_gizza(ref_path, target_path, giza_folder):
    subprocess.call(f'plain2snt.out {ref_path} {target_path}', shell=True)
    reference_target = f'{giza_folder}/reference_target.snt'
    cooc_path = f'{giza_folder}/corp.cooc'
    ref_vcb = ref_path.replace("txt", "vcb")
    target_vcb = target_path.replace("txt", "vcb")
    subprocess.call(f'snt2cooc.out {ref_vcb} {target_vcb} {reference_target} > {cooc_path}', shell=True)
    subprocess.call(f'GIZA++ -S {ref_vcb} -T {target_vcb} -C {reference_target} -CoocurrenceFile {cooc_path} -outputpath {giza_folder}', shell=True)

def unicode(s, encoding ='utf-8', errors='strict'):
    if isinstance(s, str):
        return s
    else:
        return str(s,encoding,errors=errors)

def parseAlignment(tokens): #by Sander Canisius
    assert tokens.pop(0) == "NULL"
    while tokens.pop(0) != "})":
        pass

    while tokens:
        word = tokens.pop(0)
        assert tokens.pop(0) == "({"
        positions = []
        token = tokens.pop(0)
        while token != "})":
            positions.append(int(token))
            token = tokens.pop(0)

        yield word, positions

def get_alignment(giza_folder):
    AA3_files = glob(f'{giza_folder}/*A3.final')
    assert len(AA3_files) == 1
    stream = open(AA3_files[0], 'r')
    line = stream.readline()
    encoding=False
    while line:
        assert line.startswith("#")
        src = stream.readline().split()
        trg = []
        alignment = [None for i in range(len(src))]

        for i, (targetWord, positions) in enumerate(parseAlignment(stream.readline().split())):

            trg.append(targetWord)

            for pos in positions:
                assert alignment[pos - 1] is None
                alignment[pos - 1] = i

        if encoding:
            yield [unicode(w, encoding) for w in src], [unicode(w, encoding) for w in trg], alignment
        else:
            yield src, trg, alignment

        line = stream.readline()

def get_aligned_words(giza_folder, verse_ids, reference_bible_block_idx):
    words = []
    idx = 0
    mapping = {}
    all_words = []
    for src, trg, alignment in get_alignment(giza_folder):
        verse_id = verse_ids[idx]
        block_idxs = reference_bible_block_idx[verse_id]
        for src_idx, trg_alignment in enumerate(alignment):
            all_words.append(src[src_idx])
            if trg_alignment in block_idxs:
                target_word = trg[trg_alignment]
                source_word = src[src_idx]
                if target_word not in mapping:
                    mapping[target_word] = []
                # print(f'{verse_id} {source_word} -> {target_word} (FLAGGED)')
                mapping[target_word].append(source_word)
            elif trg_alignment is None:
                # print(f'{verse_id} {src[src_idx]} -> None')
                words.append(src[src_idx])
            else:
                # print(f'{verse_id} {src[src_idx]} -> {trg[trg_alignment]}')
                words.append(src[src_idx])
        idx += 1

    block_list = []
    for key, word_list in mapping.items():
        value = pd.Series(word_list).value_counts().index[0]
        # print(f'{key} -> {value}')
        block_list.append(value)

    return block_list, all_words
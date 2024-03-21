import re

import tempfile

import os

from vocabtest import package_dir


def character_to_letter(character, language_iso):
    """
    Convert a CJK character to a 'letter', i.e., Japanese to hiragana, Chinese to pinyin, Korean to hangul
    :param character: character to convert
    :param language_iso: language iso code
    :return: converted character
    """
    import jamotools  # Korean
    from pypinyin import pinyin  # Chinese
    if language_iso == 'ja':
        return syllabify_japanese(character, 'hiragana')
    elif language_iso == 'zh':
        return ''.join([transcription[0] for transcription in pinyin(character)])
    elif language_iso == 'ko':
        return ''.join(jamotools.split_syllables(character))
    else:
        raise ValueError(f'Language not supported: {language_iso}')


def syllabify_japanese(word, type):
    """
    Convert a word to hiragana or katakana
    :param word: Japanese word to syllabify
    :param type: either 'katakana' or 'hiragana'
    :return: syllabified word
    """
    import pykakasi
    assert type in ['katakana', 'hiragana']
    key = 'kana' if type == 'katakana' else 'hira'
    kks = pykakasi.kakasi()
    return ''.join([w[key] for w in kks.convert(word)])

def uromanize(word):
    """
    Convert a word to uromanized form
    :param word: word to convert
    :return: uromanized word
    """
    uroman_pl = f'{package_dir}/bile/dependencies/uroman/bin/uroman.pl'
    assert os.path.exists(uroman_pl)
    iso = "xxx"
    with tempfile.NamedTemporaryFile() as tf, \
         tempfile.NamedTemporaryFile() as tf2:
        with open(tf.name, "w") as f:
            f.write("\n".join([word]))
        cmd = f"perl " + uroman_pl
        cmd += f" -l {iso} "
        cmd +=  f" < {tf.name} > {tf2.name}"
        os.system(cmd)
        outtexts = []
        with open(tf2.name) as f:
            for line in f:
                line = re.sub(r"\s+", " ", line).strip()
                outtexts.append(line)
        outtext = outtexts[0]
    return outtext
# 📖 VocabTest
#### Vocabulary tests for an open-ended number of languages
**TD;DR**: Vocabulary tests are useful to assess language proficiency. This repository allows create vocabulary. 📝 [Try your vocabulary knowledge](https://vocabtest.org)
*********


![example items from WikiVocab](static/wikivocab-preview.gif)

## Setup
Clone the repository and install the requirements:
```shell
git clone https://github.com/polvanrijn/VocabTest
cd VocabTest
REPO_DIR=$(pwd)
python3.9 -m venv env  # Setup virtual environment, I used Python 3.9.18 on MacOS
pip install -r requirements.txt
pip install -e .
```

### Developer requirements

<details>
<summary><b>Optionally: Install dictionaries</b></summary>

Make sure you either have `hunspell` or `myspell` installed.
```shell
DIR_DICT = ~/.config/enchant/hunspell # if you use hunspell
DIR_DICT = ~/.config/enchant/myspell # if you use myspell
mkdir -p $DIR_DICT
```

Download the Libreoffice dictionaries:
```shell
cd $DIR_DICT
git clone https://github.com/LibreOffice/dictionaries
find dictionaries/ -type f -name "*.dic" -exec mv -i {} .  \;
find dictionaries/ -type f -name "*.aff" -exec mv -i {} .  \;
rm -Rf dictionaries/
```

Manually install missing dictionaries:
```shell
# Manually install dictionaries
function get_dictionary() {
  f="$(basename -- $1)"
  wget $1 --no-check-certificate
  unzip $f "*.dic" "*.aff"
  rm -f $f
}

# Urdu
get_dictionary https://versaweb.dl.sourceforge.net/project/aoo-extensions/2536/1/dict-ur.oxt

# Western Armenian
get_dictionary https://master.dl.sourceforge.net/project/aoo-extensions/4841/0/hy_am_western-1.0.oxt

# Galician
get_dictionary https://extensions.libreoffice.org/assets/do wnloads/z/corrector-18-07-para-galego.oxt

# Welsh
get_dictionary https://master.dl.sourceforge.net/project/aoo-extensions/1583/1/geiriadur-cy.oxt
mv dictionaries/* .
rm -Rf dictionaries/

# Belarusian
get_dictionary https://extensions.libreoffice.org/assets/downloads/z/dict-be-0-58.oxt

# Marathi
get_dictionary https://extensions.libreoffice.org/assets/downloads/73/1662621066/mr_IN-v8.oxt
mv dicts/* .
rm -Rf dicts/
```

Check all dictionaries are installed:
```shell
python3 -c "import enchant
broker = enchant.Broker()
print(sorted(list(set([lang.split('_')[0] for lang in broker.list_languages()]))))"
```
</details>

<details>
<summary><b>Optionally: Install FastText</b></summary>

```shell
cd $REPO_DIR
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip3 install .
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
cd ..
```
</details>

<details>
<summary><b>Optionally: Install local UDPipe</b></summary>

Install tensorflow:
```shell
pip install tensorflow
```

Make sure GPU is available:
```shell
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Install UDPipe:
```shell
cd $REPO_DIR
git clone https://github.com/ufal/udpipe
cd udpipe
git checkout udpipe-2
git clone https://github.com/ufal/wembedding_service
pip install .
```

Download the models
```shell
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4804{/udpipe2-ud-2.10-220711.tar.gz}
tar -xvf udpipe2-ud-2.10-220711.tar.gz
rm udpipe2-ud-2.10-220711.tar.gz
```

I had to make one change to the code to make it work locally. Change line 375 in `udpipe2_server.py` to:
```python
if not hasattr(socket, 'SO_REUSEPORT'):
     socket.SO_REUSEPORT = 15
```
</details>

<details>
<summary><b>Optionally: Word alignment</b></summary>

```{shell}
cd $REPO_DIR/vocabtest/bible/
mkdir dependencies
cd dependencies
git clone https://github.com/clab/fast_align
cd fast_align
mkdir build
cd build
cmake ..
make
```
</details>

<details>
<summary><b>Optionally: Uromanize</b></summary>

```bash
cd vocabtest/vocabtest/bible/dependencies/
git clone https://github.com/isi-nlp/uroman
```
</details>

## Tests

### WikiVocab: Validated vocabulary test for 60 languages
1. Afrikaans 🇿🇦
1. Arabic (many countries)
1. Belarussian 🇧🇾
1. Bulgarian 🇧🇬
1. Catalan 🇪🇸
1. Czech 🇨🇿
1. Welsh 🇬🇧
1. Danish 🇩🇰
1. German 🇩🇪🇨🇭🇦🇹
1. Greek 🇬🇷
1. English (many countries)
1. Spanish (many countries)
1. Estionian 🇪🇪
1. Basque 🇪🇸
1. Persian 🇮🇷🇦🇫🇹🇯
1. Finnish 🇫🇮
1. Faroese 🇩🇰
1. French (many countries)
1. Irish 🇮🇪
1. Gaelic (Scottish) 🏴󠁧󠁢󠁳󠁣󠁴󠁿
1. Galician 🇪🇸
1. Gothic (dead)
1. Hebrew 🇮🇱
1. Hindi 🇮🇳
1. Croatian 🇭🇷
1. Hungarian 🇭🇺
1. Armenian 🇦🇲
1. Western Armenian
1. Indonesia 🇮🇩
1. Icelandic 🇮🇸
1. Italian 🇮🇹
1. Japanese 🇯🇵
1. Korean 🇰🇷
1. Latin (dead)
1. Lithuania 🇱🇹
1. Latvian 🇱🇻
1. Marathi 🇮🇳
1. Maltese 🇲🇹
1. Dutch 🇳🇱🇧🇪
1. Norwegian Nynorsk 🇳🇴
1. Norwegian Bokmål 🇳🇴
1. Polish 🇵🇱
1. Portuguese 🇵🇹
1. Romanian 🇷🇴
1. Russian 🇷🇺
1. Sanskrit 🇮🇳
1. Northern Sami 🇳🇴
1. Slovak 🇸🇰
1. Slovenian 🇸🇮
1. Serbian 🇷🇸
1. Swedish 🇸🇪
1. Tamil 🇮🇳🇱🇰🇸🇬
1. Telugu 🇮🇳
1. Turkish 🇹🇷
1. Uyghur 🇨🇳
1. Ukranian 🇺🇦
1. Urdu 🇵🇰🇮🇳
1. Vietnamese 🇻🇳
1. Wolof 🇸🇳
1. Chinese 🇨🇳

### BibleVocab: Vocabulary test for more than 2000 languages


## Create your own vocabulary test
Creating your own vocabulary test is easy. The only thing you need is a large amount of text in a language and need to implement two functions:
- `vocabtest.<your_dataset>.download`: which downloads the dataset and stores it in a subfolder called `data`
- `vocabtest.<your_dataset>.filter`: which filters and cleans the dataset and stores the following files in the `database` subfolder:
    - `{language_id}-filtered.csv` is a table with `word` and `count` of all words that pass the filter,
    - `{language_id}-clean.txt` is text file with all words that are cleaned, which is used for training the compound word splitter,
    - `{language_id}-all.txt` is text file with all words occurring in the corpus, which is used to reject pseudowords which are already in the corpus

You can now run your vocabulary test with:
```shell
vocabtest download <your_dataset> <language_id>
vocabtest filter <your_dataset> <language_id>
vocabtest create-pseudowords <your_dataset> <language_id>
vocabtest create-test <your_dataset> <language_id>
```


### Citation
```
@misc{vanrijn2023wikivocab,
      title={Around the world in 60 words: A generative vocabulary test for online research}, 
      author={Pol van Rijn and Yue Sun and Harin Lee and Raja Marjieh and Ilia Sucholutsky and Francesca Lanzarini and Elisabeth André and Nori Jacoby},
      year={2023},
      eprint={2302.01614},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2302.01614}, 
}
```
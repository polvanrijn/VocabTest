# 📖 VocabTest
#### Vocabulary tests for an open-ended number of 60 languages (here ~2k)
**TD;DR**: Vocabulary tests are useful to assess if someone is a native speaker of a language. This repository allows to quickly test vocabulary knowledge for 60 lanugages. 📝 [Try the test yourself](https://polvanrijn.github.io/WikiVocab/index.html)
*********


## Setup
Clone the repository and install the requirements:
```shell
git clone https://github.com/polvanrijn/VocabTest
cd VocabTest
REPO_DIR=$(pwd)
pip install -r requirements.txt
cd dependencies
DEPENDENCIES_DIR=$(pwd)
```

### Install dependencies
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
cd $DEPENDENCIES_DIR
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

## Running
The full pipeline consists of three main parts:
- Extract and collect text
- Create and filter vocabulary
- Create pseudowords

### Extract and collect texts
Here we use the longest Wikipedia articles from a given language, but you could have taken any other large text corpus (👉 read the paper to find out why we took Wikipedia). Each token is lemmatized and POS-tagged using UDPipe 2. For spellchecking we used FastText and manually installed dictionaries.

```shell
LANG=de # language code
python download_articles.py --language $LANG
python process_articles.py --language $LANG
```

### Create and filter vocabulary
Each processed article is stored as a single json file. We first merge the single processed articles into one file (our vocabulary). We now apply various filter the vocabulary, e.g. only include spell-checked nouns (see paper for all details).

```shell
python create_vocabulary.py --language $LANG
python filter_vocabulary.py --language $LANG
```

### Create pseudowords
From the curated vocabulary we can compute conditional probabilities of the n-grams (default: 5-grams). We can now sample pseudowords from the conditional probabilities of the n-grams. Once a pseudoword is created, we apply some filters. For example, the pseudoword may not occur in the vocabulary. The sampling speed depends on the language (e.g., some languages tend to mainly produce pseudowords which are already in the vocabulary). We stop the sampling after 8 hours, which gives sufficient (> 500) pseudowords. We pair each pseudoword with its closest real word.

```shell
python create_pseudowords.py --language $LANG # default is 5-grams
python pair_pseudowords.py --language $LANG
```


### Citation
```
# TODO
```

### TODOs
- [ ] Set MANIFEST.in
- [ ] Update READMEs
# BibleVocab

Steps to recreate BibleVocab. First list all bibles available bible.com:

```
python list_available_bibles.py
```

Now download all bibles:

```
python download_bibles.py
```

Compute POS tags of reference bible (mainly interested in the named entities):

```
python compute_pos_reference_bible.py
```

Match the named entities in the reference bible with the named entities in the other bibles:

```
python match_propnouns_bible.py
```
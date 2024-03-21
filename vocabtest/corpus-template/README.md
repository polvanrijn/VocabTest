# Template for a new text corpus

We assume the following folder structure:
- `data/` contains the raw text corpus
- `databases/` contains the processed text corpus
- `pseudowords/` contains the 1000 generated pseudowords
- `pairs/` contains the 500 word-pseudoword pairs
- `README.md` contains the description of the text corpus
- `download.py` downloads the raw text corpus
- `process.py` processes the raw text corpus to `databases/<language-iso>.csv`
- `filter.py` filters bad items from `databases/<language-iso>.csv` and stores it in a standardized format in `databases/<language-iso>-clean.csv`
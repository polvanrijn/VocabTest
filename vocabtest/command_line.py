import os.path
import importlib

from vocabtest.create_pseudowords import create_pseudowords
from vocabtest.create_test import create_test

import click


@click.group()
def vocabtest():
    pass

def validate_scripts(dataset, name):
    folder = os.path.dirname(os.path.abspath(__file__)) + f"/{dataset}"
    assert os.path.exists(folder), f"Folder {folder} does not exist."
    script = f"{folder}/{name}.py"
    assert os.path.exists(script), f"Script {script} does not exist."


@vocabtest.command("download")
@click.argument("dataset", required=True)
@click.argument("language", required=True)
@click.pass_context
def vocabtest_download(ctx, dataset, language, **kwargs):
    validate_scripts(dataset, "download")
    download = importlib.import_module(f"vocabtest.{dataset}.download")
    download.download(language)


@vocabtest.command("filter")
@click.argument("dataset", required=True)
@click.argument("language", required=True)
@click.pass_context
def vocabtest_filter(ctx, dataset, language, **kwargs):
    validate_scripts(dataset, "filter")
    filter = importlib.import_module(f"vocabtest.{dataset}.filter")
    filter.filter(language)


@vocabtest.command("create-pseudowords")
@click.argument("dataset", required=True)
@click.argument("language", required=True)
@click.option("--n_gram_len", default=5, type=int, help='N in N-gram')
@click.option("--max_tries", default=10 ** 6, type=int, help='Maximum number of tries')
@click.option("--n_pseudowords", default=1000, type=int, help='Number of pseudowords to generate')
@click.option("--verbose", is_flag=True, help='Verbose')
@click.pass_context
def vocabtest_create_pseudowords(ctx, dataset, language, n_gram_len, max_tries, n_pseudowords, verbose):
    create_pseudowords(dataset, language, n_gram_len, max_tries, n_pseudowords, verbose)



@vocabtest.command("create-test")
@click.argument("dataset", required=True)
@click.argument("language", required=True)
@click.option("--mean", default=-5, type=float, help='Mean percentile of words to include')
@click.option("--std", default=0.88, type=float, help='Std percentile of words to include')
@click.option("--n_pseudowords", default=500, type=int, help='Number of pseudoword pairs')
@click.pass_context
def vocabtest_create_pseudowords(ctx, dataset, language, mean, std, n_pseudowords):
    create_test(dataset, language, n_pseudowords, mean, std)



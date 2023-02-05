import click
import os
import pandas as pd

from src.models.models_utils import DBERTClassifier, RNNClassifier


def get_train_test_data(input_train_data_filepath, input_test_data_filepath):
    train_df = pd.read_csv(input_train_data_filepath)
    train_tokens = train_df['texts']
    train_labels = train_df['labels']
    if input_test_data_filepath:
        test_df = pd.read_csv(input_test_data_filepath)
        test_tokens = test_df['texts']
        test_labels = test_df['labels']
    else:
        test_tokens = None
        test_labels = None
    return train_tokens, train_labels, test_tokens, test_labels


def make_filepath(folderpath, filename):
    if folderpath != 'None':
        filepath = os.path.join(folderpath, filename)
    else:
        filepath = None
    return filepath


def train_dbert(output_filepath, input_train_data_filepath, input_test_data_filepath):
    train_tokens, train_labels, test_tokens, test_labels = get_train_test_data(
        input_train_data_filepath,
        input_test_data_filepath
    )
    clf = DBERTClassifier()
    clf.fit(train_tokens, train_labels, test_tokens, test_labels)
    clf.save_model(output_filepath)


def train_rnn(output_filepath, input_train_data_filepath, input_test_data_filepath):
    train_tokens, train_labels, test_tokens, test_labels = get_train_test_data(
        input_train_data_filepath,
        input_test_data_filepath
    )
    clf = RNNClassifier(bidirectional=True)
    clf.fit(train_tokens, train_labels, test_tokens, test_labels)
    clf.save_model(output_filepath)


def train_nationality(output_folderpath, input_train_data_folderpath, input_test_data_folderpath):
    output_filepath = os.path.join(output_folderpath, 'nationality.pkl')
    input_train_data_filepath = make_filepath(input_train_data_folderpath, 'nationality.csv')
    input_test_data_filepath = make_filepath(input_test_data_folderpath, 'nationality.csv')
    train_dbert(output_filepath, input_train_data_filepath, input_test_data_filepath)


def train_families(output_folderpath, input_train_data_folderpath, input_test_data_folderpath):
    output_filepath = os.path.join(output_folderpath, 'families.pkl')
    input_train_data_filepath = make_filepath(input_train_data_folderpath, 'families.csv')
    input_test_data_filepath = make_filepath(input_test_data_folderpath, 'families.csv')
    train_dbert(output_filepath, input_train_data_filepath, input_test_data_filepath)


def train_sex(output_folderpath, input_train_data_folderpath, input_test_data_folderpath):
    output_filepath = os.path.join(output_folderpath, 'sex.pkl')
    input_train_data_filepath = make_filepath(input_train_data_folderpath, 'sex.csv')
    input_test_data_filepath = make_filepath(input_test_data_folderpath, 'sex.csv')
    train_dbert(output_filepath, input_train_data_filepath, input_test_data_filepath)


def train_limit(output_folderpath, input_train_data_folderpath, input_test_data_folderpath):
    output_filepath = os.path.join(output_folderpath, 'limit.pkl')
    input_train_data_filepath = make_filepath(input_train_data_folderpath, 'limit.csv')
    input_test_data_filepath = make_filepath(input_test_data_folderpath, 'limit.csv')
    train_rnn(output_filepath, input_train_data_filepath, input_test_data_filepath)


@click.command()
@click.argument("output_folderpath", type=click.Path(exists=True))
@click.argument("input_train_data_folderpath", type=click.Path())
@click.argument("input_test_data_folderpath", type=click.Path(), default='None')
def train_models(output_folderpath, input_train_data_folderpath, input_test_data_folderpath):
    train_nationality(output_folderpath, input_train_data_folderpath, input_test_data_folderpath)
    train_families(output_folderpath, input_train_data_folderpath, input_test_data_folderpath)
    train_sex(output_folderpath, input_train_data_folderpath, input_test_data_folderpath)
    train_limit(output_folderpath, input_train_data_folderpath, input_test_data_folderpath)


if __name__ == "__main__":
    train_models()


"""
python -m src.models.train_models models/weights/test data/processed/train
"""


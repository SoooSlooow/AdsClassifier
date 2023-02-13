from ast import literal_eval

import click
import pandas as pd
import torch

from src.models.models_utils import AdClassifier


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("tokens_filepath", type=click.Path(exists=True))
@click.argument("output_save_filepath", type=click.Path())
def predict_labels(model_filepath, tokens_filepath, output_save_filepath):
    df = pd.read_csv(tokens_filepath)
    tokens = list(df['texts'].apply(literal_eval))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(model_filepath, 'rb') as file:
        clf = torch.load(file, map_location=device)

    predictions = clf.predict_labels(tokens)
    predictions_df = pd.DataFrame(predictions)
    output_df = pd.concat((df['texts'], predictions_df))
    output_df.to_csv(output_save_filepath, index=False)


if __name__ == '__main__':
    predict_labels()

'''
python -m src.models.predict_labels models/nnets/nnet_train.pt data/processed/test/nationality.csv models/predictions/test_nationality_predictions.csv
'''
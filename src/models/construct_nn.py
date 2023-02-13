import click

from src.models.models_utils import AdClassifier


@click.command()
@click.argument("weights_folderpath", type=click.Path(exists=True))
@click.argument("dict_filepath", type=click.Path(exists=True))
@click.argument("output_save_filepath", type=click.Path())
def construct_nn(weights_folderpath, dict_filepath, output_save_filepath):
    clf = AdClassifier(weights_folderpath, dict_filepath)
    clf.save_model(output_save_filepath)


if __name__ == "__main__":
    construct_nn()

'''
python -m src.models.construct_nn models/weights/train_weights_pt models/word_to_idx_dict.pkl models/nnets/nnet_train.pkl
python -m src.models.construct_nn models/weights/full_weights_pt models/word_to_idx_dict.pkl models/nnets/nnet_full.pkl
'''



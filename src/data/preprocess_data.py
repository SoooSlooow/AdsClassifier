import string

import click
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import pymorphy2
from sklearn.model_selection import train_test_split


class DataPreprocessor:

    def __init__(self):
        nltk.download('stopwords')
        self.morph = pymorphy2.MorphAnalyzer()
        self.tokenizer = WordPunctTokenizer()
        self.punctuation = set(string.punctuation)
        self.stopwords_russian = stopwords.words("russian")
        self.stop_tokens = (set(self.stopwords_russian) - {'и', 'или', 'не'}).union(self.punctuation)

    def tokenize_data(self, texts):
        tokens = [self.tokenizer.tokenize(str(text).lower()) for text in texts]
        return tokens

    def lemmatize_tokens_string(self, tokens_string):
        new_tokens = []
        for token in tokens_string:
            if token not in self.stop_tokens:
                new_tokens.append(self.morph.parse(token)[0].normal_form)
        return new_tokens

    def lemmatize_tokens(self, tokens):
        for i in range(len(tokens)):
            tokens[i] = self.lemmatize_tokens_string(tokens[i])

    def preprocess_texts(self, texts):
        tokens = self.tokenize_data(texts)
        self.lemmatize_tokens(tokens)
        return tokens


@click.command()
@click.argument("input_filepath_df", type=click.Path(exists=True))
@click.argument("label_name")
@click.argument("output_filepath_train_df", type=click.Path())
@click.argument("output_filepath_test_df", type=click.Path())
def preprocess_data(input_filepath_df, label_name,
                    output_filepath_train_df, output_filepath_test_df):
    preprocessor = DataPreprocessor()
    df = pd.read_csv(input_filepath_df)
    preprocessed_df = pd.DataFrame()
    preprocessed_df['texts'] = preprocessor.preprocess_texts(df['texts'])
    preprocessed_df['labels'] = df[label_name]
    train_df, test_df = train_test_split(preprocessed_df, test_size=0.2,
                                         random_state=42, shuffle=True,
                                         stratify=preprocessed_df['labels'])
    train_df.to_csv(output_filepath_train_df, index=False)
    test_df.to_csv(output_filepath_test_df, index=False)


if __name__ == "__main__":
    preprocess_data()

# python -m src.data.preprocess_data data/raw/labeled_texts_fixed_binary.csv nationality data/processed/train_nationality.csv data/processed/test_nationality.csv

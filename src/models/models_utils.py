import os
import pickle
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve
import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel


def get_roc_aucs(y, probas):
    y_onehot = pd.get_dummies(y)
    roc_auc_scores = []
    if y_onehot.shape[1] > 2:
        for i in range(y_onehot.shape[1]):
            roc_auc_scores.append(roc_auc_score(y_onehot[i], probas[:, i]))
        roc_auc_scores.append(roc_auc_score(y, probas, multi_class='ovo', average='macro'))
    else:
        roc_auc_scores.append(roc_auc_score(y, probas[:, 1]))
    return roc_auc_scores


def get_max_f1_score(y, probas):
    if probas.shape[1] != 2:
        raise ValueError('Expected probabilities for 2 classes would be given')
    y_onehot = pd.get_dummies(y)
    f1_score = []
    threshold = []
    p, r, t = precision_recall_curve(y, probas[:, 1])
    f1_scores = 2 * p * r / (p + r + 0.001)
    threshold.append(t[np.argmax(f1_scores)])
    f1_score.append(np.max(f1_scores))
    return f1_score, threshold


class RNN(nn.Module):

    def __init__(self, vectors, n_of_words, n_of_classes, num_layers, bidirectional):
        dim = vectors.shape[1]
        d = 2 if bidirectional else 1
        super().__init__()
        self.emb = nn.Embedding(n_of_words, dim)
        self.emb.load_state_dict({'weight': torch.tensor(vectors)})
        self.emb.weight.requires_grad = False
        self.gru = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True,
                          num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(dim * num_layers * d, n_of_classes)

    def forward(self, batch):
        emb = self.emb(batch)
        _, last_state = self.gru(emb)
        last_state = torch.permute(last_state, (1, 0, 2)).reshape(1, batch.shape[0], -1).squeeze()
        out = self.linear(last_state.squeeze())
        if len(out.size()) == 1:
            out = out.unsqueeze(0)
        return out


class DistilBERTClass(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.l1 = DistilBertModel.from_pretrained('DeepPavlov/distilrubert-small-cased-conversational')
        self.linear = torch.nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        output = self.linear(pooler)
        return output


class BaseClassifier:

    def __init__(self, batch_size=16, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def preprocess_with_random_initialization(self, train_tokens):
        self.pad_idx = 0
        self.unk_idx = 1

        set_of_words = set()
        for tokens_string in train_tokens:
            set_of_words.update(tokens_string)

        self.idx_to_word = ['PADDING', 'UNK'] + list(set_of_words)
        self.word_to_idx = {key: i for i, key in enumerate(self.idx_to_word)}
        self.amount_of_words = len(self.idx_to_word)

        self.vectors = np.zeros((len(self.idx_to_word), 300))
        self.vectors[0, :] = np.zeros(300)
        self.vectors[1:len(self.idx_to_word), :] = (np.random.rand(len(self.idx_to_word) - 1, 300) - 0.5) / 300

    def preprocess(self, vectors_file_path):
        self.emb = KeyedVectors.load_word2vec_format(vectors_file_path)

        self.pad_idx = 0
        self.unk_idx = 1

        self.idx_to_word = ['PADDING', 'UNK'] + list(self.emb.index_to_key)
        self.word_to_idx = {key: i for i, key in enumerate(self.idx_to_word)}
        self.amount_of_words = len(self.idx_to_word)

        self.vectors = np.zeros((len(self.idx_to_word), 300))
        self.vectors[0, :] = np.zeros(300)
        self.vectors[1, :] = (np.random.rand(300) - 0.5) / 300
        for i in range(2, len(self.idx_to_word)):
            self.vectors[i, :] = self.emb.get_vector(self.idx_to_word[i])

    def fit(self, train_tokens, y_train, test_tokens=None, y_test=None,
            reinitialize=True, stop_epochs=None, show_logs=False):
        if reinitialize:
            self.n_of_classes = y_train.nunique()
            self.initialize_nnet()

        self.print_test = test_tokens and y_test
        self.stop_epochs = stop_epochs
        train_scores = []
        self.train_scores_mean = []
        self.test_scores = []
        self.test_aucs = []
        self.test_f1 = []
        criterion = nn.CrossEntropyLoss()
        for epoch in tqdm.tqdm(range(self.epochs)):
            self.epoch = epoch
            self.nnet.train()
            train_batches = self.batch_generator(train_tokens, y_train)
            test_batches = self.batch_generator(test_tokens, y_test)
            for i, batch in tqdm.tqdm(
                    enumerate(train_batches),
                    total=len(train_tokens) // self.batch_size
            ):
                pred = self.nnet(batch['tokens'])
                loss = criterion(pred, batch['labels'])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if show_logs and i % 400 == 0:
                    train_score = criterion(self.nnet(batch['tokens']), batch['labels'])
                    print(train_score.item())
                    train_scores.append(train_score.item())
            if show_logs:
                self.train_scores_mean.append(sum(train_scores) / len(train_scores))
                train_scores = []
                if self.print_test:
                    test_pred_prob = torch.tensor([], device='cpu')
                    with torch.no_grad():
                        for batch in test_batches:
                            test_batch_pred_prob = self.nnet(batch['tokens'])
                            test_batch_pred_prob_cpu = test_batch_pred_prob.to('cpu')
                            test_pred_prob = torch.cat((test_pred_prob, test_batch_pred_prob_cpu), 0)
                    test_score = criterion(test_pred_prob, torch.tensor(y_test.values, device='cpu'))
                    self.test_scores.append(test_score.item())
                    test_pred_probas = F.softmax(test_pred_prob).detach().cpu().numpy()
                    self.test_aucs.append(get_roc_aucs(y_test, test_pred_probas))
                    self.test_f1.append(get_max_f1_score(y_test, test_pred_probas)[0])
                self.print_metrics()
            if self.early_stopping_check():
                break

    def count_tokens(self, tokens):
        self.words_counter = Counter()
        self.amount_of_tokens = 0
        for s in tokens:
            self.words_counter.update(s)
            self.amount_of_tokens += len(s)

    def index_tokens(self, tokens_string):
        return [self.word_to_idx.get(token, self.unk_idx) for token in tokens_string]

    def fill_with_pads(self, tokens):
        tokens = deepcopy(tokens)
        max_len = 0
        for tokens_string in tokens:
            max_len = max(max_len, len(tokens_string))
        for tokens_string in tokens:
            for i in range(len(tokens_string), max_len):
                tokens_string.append(self.pad_idx)
        return tokens

    def as_matrix(self, tokens):
        tokens = deepcopy(tokens)
        for j, s in enumerate(tokens):
            tokens[j] = self.index_tokens(s)
        tokens = self.fill_with_pads(tokens)
        return tokens

    def batch_generator(self, tokens, labels):
        for i in range(0, len(tokens), self.batch_size):
            batch_tokens = tokens[i: i + self.batch_size]
            batch_labels = torch.tensor(labels.values[i: i + self.batch_size],
                                        dtype=torch.long,
                                        device=self.device)

            batch_tokens_idx = torch.tensor(self.as_matrix(batch_tokens),
                                            dtype=torch.int,
                                            device=self.device)
            if len(batch_tokens_idx.size()) == 1:
                batch_tokens_idx = torch.unsqueeze(batch_tokens_idx, 0)

            batch = {
                'tokens': batch_tokens_idx,
                'labels': batch_labels
            }
            yield batch

    def print_metrics(self, print_test=True):

        if self.print_test:
            print(f'epoch {self.epoch}/{self.epochs}')
            print('auc', self.test_aucs[-1])
            print('score', self.test_scores[-1])
            print('f1 score', self.test_f1[-1])

            legend_labels = []
            if self.n_of_classes > 2:
                for i in range(self.n_of_classes):
                    legend_labels.append(f'Class {i}')
            legend_labels.append('General')

            plt.figure(figsize=(5, 15))

            plt.clf()

            plt.subplot(3, 1, 1)
            plt.plot(np.arange(1, self.epoch + 2), self.test_aucs)
            plt.grid()
            plt.title('Test ROC AUC')
            plt.xlabel('Num. of epochs')
            plt.ylabel('ROC AUC')
            plt.legend(legend_labels)

            plt.subplot(3, 1, 2)
            plt.plot(np.arange(1, self.epoch + 2), self.test_f1)
            plt.grid()
            plt.title('Test F1-score')
            plt.xlabel('Num. of epochs')
            plt.ylabel('F1-score')
            plt.legend(legend_labels)

            plt.subplot(3, 1, 3)
            plt.plot(np.arange(1, self.epoch + 2), self.train_scores_mean, label='Train loss')
            plt.plot(np.arange(1, self.epoch + 2), self.test_scores, label='Test loss')
            plt.title('Loss')
            plt.xlabel('Num. of epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            plt.draw()

        else:
            plt.figure(figsize=(5, 15))
            plt.plot(np.arange(1, self.epoch + 2), self.train_scores_mean, label='Train loss')
            plt.title('Loss')
            plt.xlabel('Num. of epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            plt.show()

    def early_stopping_check(self):
        if self.stop_epochs is None or self.stop_epochs >= len(self.test_scores):
            return False
        else:
            print(self.test_scores)
            first_score = np.array(self.test_scores)[-self.stop_epochs - 1]
            last_scores = np.array(self.test_scores)[-self.stop_epochs:]
            return np.all(last_scores >= first_score)


class RNNClassifier(BaseClassifier):

    def __init__(self, batch_size=16, epochs=100,
                 num_layers=1, bidirectional=False):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_layers = num_layers
        self.bidirectional = bidirectional

    def initialize_nnet(self):
        self.nnet = RNN(self.vectors, self.amount_of_words,
                        n_of_classes=self.n_of_classes,
                        num_layers=self.num_layers,
                        bidirectional=self.bidirectional).to(self.device)
        self.optimizer = torch.optim.Adam(self.nnet.parameters())

    def save_model(self, filepath):
        parameters = dict()
        parameters['gru'] = dict()
        parameters['linear'] = dict()
        gru_dict = self.nnet.gru.state_dict()
        for param_gru in gru_dict:
            parameters['gru'][param_gru] = gru_dict[param_gru]
        linear_dict = self.nnet.linear.state_dict()
        for param_linear in linear_dict:
            parameters['linear'][param_linear] = linear_dict[param_linear]
        with open(filepath, 'wb') as file:
            pickle.dump(parameters, file)


class DBERTClassifier(BaseClassifier):

    def __init__(self, batch_size=16, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def initialize_nnet(self):
        self.nnet = DistilBERTClass(self.n_of_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=2e-6)
        # 'DeepPavlov/rubert-base-cased' 'DeepPavlov/distilrubert-small-cased-conversational',
        self.tokenizer = DistilBertTokenizer.from_pretrained('DeepPavlov/distilrubert-small-cased-conversational',
                                                             do_lower_case=True)

    def batch_generator(self, tokens, labels):
        for i in range(0, len(tokens), self.batch_size):
            batch_tokens = tokens[i: i + self.batch_size]
            batch_tokens = [' '.join(s) for s in batch_tokens]
            batch_labels = torch.tensor(labels.values[i: i + self.batch_size],
                                        dtype=torch.long,
                                        device=self.device)
            if len(batch_tokens) == 1:
                inputs = self.tokenizer.encode_plus(
                    batch_tokens,
                    None,
                    add_special_tokens=True,
                    max_length=512,
                    truncation=True,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )
            else:
                inputs = self.tokenizer.batch_encode_plus(
                    batch_tokens,
                    add_special_tokens=True,
                    max_length=512,
                    truncation=True,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )
            batch_token_ids = torch.tensor(inputs['input_ids'], device=self.device, dtype=torch.long)
            batch_mask = torch.tensor(inputs['attention_mask'], device=self.device, dtype=torch.long)
            batch_token_type_ids = torch.tensor(inputs["token_type_ids"], device=self.device, dtype=torch.long)
            if len(batch_tokens) == 1:
                batch_token_ids = batch_token_ids.unsqueeze(0)
                batch_mask = batch_mask.unsqueeze(0)
                batch_token_type_ids = batch_token_type_ids.unsqueeze(0)
            batch = {
                'tokens': batch_token_ids,
                'mask': batch_mask,
                'token_type_ids': batch_token_type_ids,
                'labels': batch_labels
            }
            yield batch

    def fit(self, train_tokens, y_train, test_tokens=None, y_test=None,
            reinitialize=True, stop_epochs=None, show_logs=False):
        if reinitialize:
            self.n_of_classes = y_train.nunique()
            self.initialize_nnet()

        self.stop_epochs = stop_epochs
        self.print_test = test_tokens and y_test
        train_scores = []
        self.train_scores_mean = []
        self.test_scores = []
        self.test_aucs = []
        self.test_f1 = []
        criterion = nn.CrossEntropyLoss()
        for epoch in tqdm.tqdm(range(self.epochs)):
            self.epoch = epoch
            self.nnet.train()
            train_batches = self.batch_generator(train_tokens, y_train)
            test_batches = self.batch_generator(test_tokens, y_test)
            for i, batch in tqdm.tqdm(
                    enumerate(train_batches),
                    total=len(train_tokens) // self.batch_size
            ):
                pred = self.nnet(batch['tokens'], batch['mask'], batch['token_type_ids'])
                loss = criterion(pred, batch['labels'])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if show_logs and i % 400 == 0:
                    train_score = criterion(self.nnet(batch['tokens'], batch['mask'], batch['token_type_ids']),
                                            batch['labels'])
                    print(train_score.item())
                    train_scores.append(train_score.item())
            if show_logs:
                self.train_scores_mean.append(sum(train_scores) / len(train_scores))
                train_scores = []
                if self.print_test:
                    test_pred_prob = torch.tensor([], device='cpu')
                    with torch.no_grad():
                        for batch in test_batches:
                            test_batch_pred_prob = self.nnet(batch['tokens'], batch['mask'], batch['token_type_ids'])
                            test_batch_pred_prob_cpu = test_batch_pred_prob.to('cpu')
                            test_pred_prob = torch.cat((test_pred_prob, test_batch_pred_prob_cpu), 0)
                        test_score = criterion(test_pred_prob, torch.tensor(y_test.values, device='cpu'))
                        self.test_scores.append(test_score.item())
                        test_pred_probas = F.softmax(test_pred_prob).detach().cpu().numpy()
                        self.test_aucs.append(get_roc_aucs(y_test, test_pred_probas))
                        self.test_f1.append(get_max_f1_score(y_test, test_pred_probas)[0])
                    self.print_metrics()
                if self.early_stopping_check():
                    break

    def predict_proba(self, tokens, labels):
        batches = self.batch_generator(tokens, labels)
        pred_probas = torch.tensor([], device=self.device)
        with torch.no_grad():
            for batch in batches:
                batch_prob = self.nnet(batch['tokens'], batch['mask'],
                                       batch['token_type_ids'])
                pred_probas = torch.cat((pred_probas, batch_prob))
        return F.softmax(pred_probas).detach().cpu().numpy()

    def predict(self, tokens, labels):
        return np.argmax(self.predict_proba(tokens, labels), axis=1)

    def save_model(self, filepath):
        parameters = dict()
        parameters['linear'] = dict()
        linear_dict = self.nnet.linear.state_dict()
        for param_linear in linear_dict:
            parameters['linear'][param_linear] = linear_dict[param_linear]
        with open(filepath, 'wb') as file:
            pickle.dump(parameters, file)


class AdsClassifierNN(nn.Module):

    def __init__(self, weights_folder):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__()

        weights_nationality_path = os.path.join(weights_folder, 'model_nationality.pt')
        weights_families_path = os.path.join(weights_folder, 'model_families.pt')
        weights_sex_path = os.path.join(weights_folder, 'model_sex.pt')
        weights_limit_path = os.path.join(weights_folder, 'model_limit.pt')

        weights_limit = torch.load(weights_limit_path, map_location=device)
        weights_nationality = torch.load(weights_nationality_path, map_location=device)
        weights_families = torch.load(weights_families_path, map_location=device)
        weights_sex = torch.load(weights_sex_path, map_location=device)

        dim = weights_limit['emb']['weight'].shape[1]
        num_layers = 1
        bidirectional = True
        d = 2 if bidirectional else 1

        self.emb = nn.Embedding(weights_limit['emb']['weight'].shape[0], dim)
        self.emb.load_state_dict(weights_limit['emb'])
        self.gru = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True,
                          num_layers=num_layers, bidirectional=bidirectional)
        self.gru.load_state_dict(weights_limit['gru'])
        self.linear_limit = nn.Linear(dim * num_layers * d, 2)
        self.linear_limit.load_state_dict(weights_limit['linear'])

        self.dbert = DistilBertModel.from_pretrained('DeepPavlov/distilrubert-small-cased-conversational')
        self.linear_nationality = nn.Linear(768, 2)
        self.linear_nationality.load_state_dict(weights_nationality['linear'])
        self.linear_families = nn.Linear(768, 2)
        self.linear_families.load_state_dict(weights_families['linear'])
        self.linear_sex = nn.Linear(768, 2)
        self.linear_sex.load_state_dict(weights_sex['linear'])

    def forward(self, input_ids, attention_mask, input_ids_rnn):
        dbert_output = self.dbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = dbert_output[0]
        pooler = hidden_state[:, 0]
        output_nationality = self.linear_nationality(pooler)
        output_families = self.linear_families(pooler)
        output_sex = self.linear_sex(pooler)

        emb = self.emb(input_ids_rnn)
        _, last_state = self.gru(emb)
        last_state = torch.permute(last_state, (1, 0, 2)).reshape(1, input_ids.shape[0], -1).squeeze()
        output_limit = self.linear_limit(last_state.squeeze())
        if len(output_limit.size()) == 1:
            output_limit = output_limit.unsqueeze(0)

        res = {
            'nationality': output_nationality,
            'families': output_families,
            'sex': output_sex,
            'limit': output_limit
        }
        return res


class AdClassifier:

    def __init__(self, weights_folder, dictionary_path):
        self.batch_size = 16

        with open(dictionary_path, 'rb') as file:
            self.word_to_idx = pickle.load(file)

        self.unk_idx = 1
        self.pad_idx = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.nn = AdsClassifierNN(weights_folder)
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'DeepPavlov/distilrubert-small-cased-conversational',
            do_lower_case=True
        )

    def index_tokens(self, tokens_string):
        return [self.word_to_idx.get(token, self.unk_idx) for token in tokens_string]

    def fill_with_pads(self, tokens):
        tokens = deepcopy(tokens)
        max_len = 0
        for tokens_string in tokens:
            max_len = max(max_len, len(tokens_string))
        for tokens_string in tokens:
            for i in range(len(tokens_string), max_len):
                tokens_string.append(self.pad_idx)
        return tokens

    def as_matrix(self, tokens):
        tokens = deepcopy(tokens)
        for j, s in enumerate(tokens):
            tokens[j] = self.index_tokens(s)
        tokens = self.fill_with_pads(tokens)
        return tokens

    def batch_generator(self, tokens):
        for i in range(0, len(tokens), self.batch_size):
            batch_tokens = tokens[i: i + self.batch_size]
            batch_tokens = [' '.join(s) for s in batch_tokens]
            if len(batch_tokens) == 1:
                inputs = self.tokenizer.encode_plus(
                    batch_tokens,
                    None,
                    add_special_tokens=True,
                    max_length=512,
                    truncation=True,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )
            else:
                inputs = self.tokenizer.batch_encode_plus(
                    batch_tokens,
                    add_special_tokens=True,
                    max_length=512,
                    truncation=True,
                    pad_to_max_length=True,
                    return_token_type_ids=True
                )
            batch_token_ids = torch.tensor(inputs['input_ids'], device=self.device, dtype=torch.long)
            batch_mask = torch.tensor(inputs['attention_mask'], device=self.device, dtype=torch.long)
            batch_token_type_ids = torch.tensor(inputs['token_type_ids'], device=self.device, dtype=torch.long)

            batch_tokens_rnn = tokens[i: i + self.batch_size]
            batch_tokens_rnn_ids = torch.tensor(self.as_matrix(batch_tokens_rnn),
                                                dtype=torch.int,
                                                device=self.device)

            if len(batch_tokens) == 1:
                batch_token_ids = batch_token_ids.unsqueeze(0)
                batch_mask = batch_mask.unsqueeze(0)
                batch_token_type_ids = batch_token_type_ids.unsqueeze(0)
                batch_tokens_rnn_ids = torch.unsqueeze(batch_tokens_rnn_ids, 0)
            batch = {
                'tokens': batch_token_ids,
                'mask': batch_mask,
                'token_type_ids': batch_token_type_ids,
                'tokens_rnn': batch_tokens_rnn_ids
            }
            yield batch

    def predict_probas(self, tokens):
        batches = self.batch_generator(tokens)
        pred_probas = {'nationality': torch.tensor([], device=self.device),
                       'families': torch.tensor([], device=self.device),
                       'sex': torch.tensor([], device=self.device),
                       'limit': torch.tensor([], device=self.device)}
        with torch.no_grad():
            for batch in batches:
                batch_probas = self.nn(batch['tokens'], batch['mask'], batch['tokens_rnn'])
                for batch_prob_label in batch_probas:
                    pred_probas[batch_prob_label] = torch.cat((pred_probas[batch_prob_label],
                                                               batch_probas[batch_prob_label]))
                for pred_prob_label in pred_probas:
                    pred_probas[pred_prob_label] = F.softmax(pred_probas[pred_prob_label]). \
                        detach().cpu().numpy()
        return pred_probas

    def save_model(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

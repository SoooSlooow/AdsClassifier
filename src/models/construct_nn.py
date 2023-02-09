from models_utils import AdsClassifierNN, AdClassifier
import torch

folder_path = '../../models/weights/train_weights_pt'
dict_path = '../../models/word_to_idx_dict.pkl'

'''
clf = AdsClassifierNN(folder_path)

a = torch.tensor([[2, 1, 2, 3], [2, 4, 5, 6]])
b = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
c = torch.tensor([[2, 1, 2, 3], [2, 4, 5, 6]])

res = clf(a, b, c)
print(res)
'''


tokens = [['сдать', 'квартира', 'только', 'славянин', 'рф'],
          ['исключительно', 'женщина', 'или', 'девушка'],
          ['строго', 'не', 'более', '2', 'человек']]
clf = AdClassifier(folder_path, dict_path)
res = clf.predict_probas(tokens)
print(res)




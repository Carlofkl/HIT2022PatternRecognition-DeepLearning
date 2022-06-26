import pandas as pd
import jieba
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import MinMaxScaler


shopping_cats = {'书籍': 0, '平板': 1, '手机': 2, '水果': 3, '洗发水': 4, '热水器': 5, '蒙牛': 6, '衣服': 7, '计算机': 8, '酒店': 9}  # 全部类别


class Shopping(Dataset):
    def __init__(self, my_list):
        self.cats = []
        self.reviews = []
        for i in range(len(my_list)):
            self.cats.append(my_list[i]['cat'])
            self.reviews.append(my_list[i]['review'])

    def __getitem__(self, idx):
        # return {"cat": self.cats[idx], "review": self.reviews[idx]}
        return {'cat': self.cats[idx], 'review': torch.tensor(np.array(self.reviews[idx]))}

    def __len__(self):
        return len(self.cats)


class Climate(Dataset):
    def __init__(self, my_data, my_label):
        sentence = []
        for i in range(my_data.shape[0]):
            word = []
            for j in range(0, 5):
                word.append(int(my_data.iloc[i, j]))
            sentence.append(word)
        self.data = np.array(sentence)
        self.label = my_label

    def __getitem__(self, idx):
        attr = self.data[idx]
        label = float(self.label.iloc[idx])
        return {'cat': label, 'review': attr}  # 和上面保持一致

    def __len__(self):
        return len(self.label)


def build_shopping(args):
    df = pd.read_csv("./dataset/online_shopping_10_cats.csv")
    val_list = []
    test_list = []
    train_list = []

    reviews = []
    cats = []
    for index, row in df.iterrows():
        if not isinstance(row['review'], str):
            continue
        cats.append(shopping_cats[row['cat']])
        reviews.append(row['review'])
    tokens = [jieba.lcut(i) for i in reviews]  # 分词

    model = Word2Vec(tokens, min_count=1, hs=1, window=3, vector_size=args.input_size)
    reviews_vector = [[model.wv[word] for word in sentence] for sentence in tokens]  # 转换成vector的reviews

    for i in range(62773):
        if i % 5 == 4:
            val_list.append({'cat': cats[i], 'review': reviews_vector[i]})
        elif i % 5 == 0:
            test_list.append({'cat': cats[i], 'review': reviews_vector[i]})
        else:
            train_list.append({'cat': cats[i], 'review': reviews_vector[i]})

    # 因为每句句子长度不同，而且是每个句子中的单词进行训练，所以相当于batch-size只能为1
    train_loader = DataLoader(Shopping(train_list), shuffle=True, batch_size=1)
    val_loader = DataLoader(Shopping(train_list), shuffle=True, batch_size=1)
    test_loader = DataLoader(Shopping(train_list), shuffle=True, batch_size=1)

    return train_loader, val_loader, test_loader


def build_climate(args):
    # read data
    data_path = "./dataset/jena_climate_2009_2016.csv"
    dataset = pd.read_csv(data_path, parse_dates=['Date Time'], index_col=['Date Time'])
    # insert new index
    dataset['year'] = dataset.index.year
    dataset['hour'] = dataset.index.hour
    # normalize hour
    dataset['sin(h)'] = [np.sin(x * (2 * np.pi / 24)) for x in dataset['hour']]
    dataset['cos(h)'] = [np.cos(x * (2 * np.pi / 24)) for x in dataset['hour']]
    # split train and test
    train_set = dataset[dataset['year'].isin(range(2009, 2015))]
    test_set = dataset[dataset['year'].isin(range(2015, 2018))]
    # determine attributes deciding T
    attr = ['H2OC (mmol/mol)', 'rho (g/m**3)', 'sh (g/kg)', 'Tpot (K)', 'VPmax (mbar)']
    # normalize other attributes
    for col in attr:
        scaler = MinMaxScaler()
        if col not in ['sin(h)', 'cos(h)', 'T (degC)']:
            dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1))
    # get train/test data
    train_data = train_set[attr]
    train_label = train_set['T (degC)']

    test_data = test_set[attr]
    test_label = test_set['T (degC)']

    # get train/test loader
    train_loader = DataLoader(Climate(train_data, train_label), shuffle=True, batch_size=1)
    test_loader = DataLoader(Climate(test_data, test_label), shuffle=True, batch_size=1)
    return train_loader, test_loader

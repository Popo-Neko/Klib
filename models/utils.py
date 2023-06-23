from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader


def train_val_split(in_features, labels, test_size=0.2, random_seed=123):
    """
    retrun train_X, val_X, train_y, val_y
    :param in_features: features
    :param labels: labels
    :param test_size: val/(val+train) ratio default=0.2
    :param random_seed: random_seed
    :return:
    """
    train_texts, val_texts, train_labels, val_labels = train_test_split(in_features,
                                                                        labels,
                                                                        test_size=0.2,
                                                                        random_state=123)
    return train_texts, val_texts, train_labels, val_labels


class Tokenizer:
    def __init__(self, max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        assert max_length <= 512
        self.max_length = max_length

    def __call__(self, text, truncation=True, padding=True):
        encodings = self.tokenizer(text, truncation=True, padding=True, max_length=self.max_length)
        return encodings


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_loader(dataset, batch_size=16, shuffle=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


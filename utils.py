import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.tensorboard import SummaryWriter
from models.utils import Tokenizer, TextClassificationDataset, train_val_split, get_loader


def count_tags(df, count):
    """
    :param df: target-dataframe
    :param count: 'kinds' , 'kinds-freq'
    :return: 'kinds' return kinds list; 'freq' return kinds-freq Counter
    """
    if count == 'kinds':
        s1 = set()
        for i in tqdm(df['questionTags']):
            tags = i.split(',')
            for j in tags:
                s1.add(j)
        return list(s1)
    elif count == 'freq':
        s1 = []
        for i in tqdm(df['questionTags']):
            tags = i.split(',')
            for j in tags:
                s1.append(j)
                print(j)
        counter = Counter(s1)
        return counter


def contain_keyword(str1, keywords):
    """
    check whether str1 has keywords
    :param str1:
    :param keywords:
    :return:
    """
    for _ in keywords:
        if _ in str1:
            return True
    return False


def read_dataset(root, train=True, column_name='combinedText', type_name='type', tensor_type='pt',
                 tokenizer_name='bert-base-chinese', test_size=0.2, random_seed=1):
    """
    read dataset，convert csv to X_train, y_train or X_test, y_test
    :param random_seed: set a randomn seed numebr
    :param test_size: test data ratio
    :param tokenizer_name: tokenizer to choose from hugging-face
    :param type_name: the name of label data's name in csv
    :param root: csv path
    :param train: train data or test data
    :param column_name: the name of text data's column in csv
    :param tensor_type: pt or tf etc.
    :return: [tensor+label]
    """
    # 计算最长字符长度
    df = pd.read_csv(root)
    try:
        text_list = list(df[column_name])
    except KeyError:
        raise KeyError('The column name of text is not default "combinedText". Input your column name or '
                       'change the column name into "combinedText"')
    text_lengths = [len(text) for text in text_list]
    # 选择一个适当的百分位数（例如，90%）或固定的值作为最大长度
    max_length = int(np.percentile(text_lengths, 90))  # 或者直接指定一个固定值
    # 加载预训练的模型 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, from_cache=True)
    texts = df[column_name]
    x = []
    # 使用 tokenizer 将文本转换为张量表示，进行填充和截断操作（未设置 max_length）
    for i in tqdm(texts, desc=f'Tokenizer in Working(length={max_length})'):
        inputs_ids = tokenizer.encode(i, padding='max_length',
                                      truncation=True, return_tensors=tensor_type, max_length=max_length)
        x.append(inputs_ids)
    y = list(df[type_name])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_seed)
    if train:
        return x_train, y_train
    else:
        return x_test, y_test


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    if y_hat.size() != y.size():
        raise ValueError("y_hat and y have different shape")
    size = y.size(0)
    correct_predictions = torch.eq(y_hat, y).sum().item()
    return float(correct_predictions/size)


def load_checkpoint(model, checkpoint_path):
    checkpoint_file = checkpoint_path
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def train(train_dataset, val_dataset, model, batch_size, num_epochs, checkpoint=False,
          step=100, learning_rate=1e-5, save4step=10):
    # DataLoader
    train_loader = get_loader(train_dataset, batch_size)
    val_loader = get_loader(val_dataset, batch_size)

    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Model
    if checkpoint is not False:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint)
    model.to(device)
    model.train()

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # SummaryWriter
    datetime_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logs_path = "../logs/" + datetime_str
    writer1 = SummaryWriter(f"{logs_path}/train_loss")
    writer2 = SummaryWriter(f"{logs_path}/test_loss")
    writer3 = SummaryWriter(f"{logs_path}/train_acc")
    writer4 = SummaryWriter(f"{logs_path}/test_acc")

    # train_loop
    for epoch in range(num_epochs):
        model.train()
        for i, batch in tqdm(enumerate(train_loader), total=int(len(train_dataset)/batch_size)):
            if i % save4step == 0:
                # every number of steps for checkpoint
                checkpoint_name = f'model_checkpoint_{i}.pth'
                torch.save(model.state_dict(), os.path.join(r'../models/checkpoints', checkpoint_name))
            if i > step:
                break
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            with torch.no_grad():
                logits = outputs.logits
                train_acc = accuracy(logits, labels)
                writer1.add_scalar("loss-step-train", loss, i)
                writer3.add_scalar("acc-step-train", train_acc, i)
        with torch.no_grad():
            model.eval()
            for i, batch in tqdm(enumerate(val_loader)):
                if i > step:
                    break
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                test_acc = accuracy(logits, labels)
                writer2.add_scalar("loss-step-test", loss, i)
                writer4.add_scalar("acc-step-test", test_acc, i)
    writer1.close()
    writer2.close()
    writer3.close()
    writer4.close()


# 6.inference
def inference(test_dataset, model, checkpoint_file_path, batch_size, shuffle=True, mode='acc'):
    # loader
    test_loader = get_loader(test_dataset, batch_size, shuffle)

    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    model = load_checkpoint(model, checkpoint_file_path)
    model.to(device)

    # inference
    if mode == 'acc':
        predictions, targets= [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, total=int(len(test_dataset)/batch_size)):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=1)
                predictions.extend(predicted_labels.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        predictions = torch.tensor(predictions)
        targets = torch.tensor(targets)
        acc = accuracy(predictions, targets)
        return acc
    elif mode == 'output':
        predictions = []
        with torch.no_grad():
            for batch in tqdm(test_loader, total=int(len(test_dataset)/batch_size)):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=1)
                predictions.extend(predicted_labels.cpu().numpy())
        return predictions
    else:
        raise ValueError("mode should be 'acc' or 'output' ! ")

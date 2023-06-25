from utils.utils import Tokenizer, TextClassificationDataset, train_val_split
import pandas as pd

# 1. 准备数据集
train_data_path = "../data/train_data4type.csv"
test_data_path = "../data/test_data4type.csv"
df_train = pd.read_csv(train_data_path, encoding='utf_8_sig')
df_test = pd.read_csv(test_data_path, encoding='utf_8_sig')
train_X, train_y = list(df_train['combinedText']), list(df_train['type'])
test_X, test_y = list(df_test['combinedText']), list(df_test['type'])

# 2. tokenizer and encodings
tokenizer = Tokenizer()
train_text, val_text, train_labels, val_labels = train_val_split(train_X, train_y)
test_text, _, test_labels, _ = train_val_split(test_X, test_y, test_size=0)
train_encodings = tokenizer(train_text)
val_encodings = tokenizer(val_text)
test_encodings = tokenizer(test_text)

# 3. dataset
train_dataset = TextClassificationDataset(encodings=train_encodings, labels=train_labels)
val_dataset = TextClassificationDataset(encodings=val_encodings, labels=val_labels)
test_dataset = TextClassificationDataset(encodings=test_encodings, labels=test_labels)

# 4. model
# model = Model()


if __name__ == '__main__':
    print("train")


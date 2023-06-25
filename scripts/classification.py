import io
import os

import chardet
import pandas as pd

from models.model import Model
from utils.utils import Tokenizer, TextClassificationDataset
from utils.utils import inference, contain_keyword
from data.key_words import Subtitle


# 分类分为三阶段
# 一阶段为所处时期的分类(孕前和产后)，为互斥分类，采用深度学习模型
# 二阶段和三阶段为副标题和标签的分类，为非互斥分类，采用关键词提取
# 分类顺序：一阶段->二阶段->三阶段


def input_check(file):
    """
    check if bytes file or file path, return dataframe
    :param file: bytes file or file path
    :return: dataframe
    """
    if type(file) == str:
        with open(file, mode='rb') as f:
            content = f.read()
        result = chardet.detect(content)
        df = pd.read_csv(file, encoding="ansi" if result['encoding'][:2] == "GB" else result['encoding'])
    elif type(file) == bytes:
        result = chardet.detect(file)
        file = io.BytesIO(file)
        df = pd.read_csv(file, encoding="ansi" if result['encoding'][:2] == "GB" else result['encoding'])
    else:
        raise TypeError("filepath parameter must be a path of a data table or a bytes object of data table")
    return df


def output_check(df, result_path, mode):
    if mode == 'file':
        if result_path is not None:
            output_path = result_path
            df.to_csv(output_path, encoding='utf-8-sig', index=False)
            return f"Output was saved in {output_path}\n Ctrl + left click to preview."
        else:
            output_path = os.path.join(os.getcwd(), r"./temp_files/output.csv")
            df.to_csv(output_path, encoding='utf-8-sig', index=False)
            return output_path
    elif mode == 'data':
        return df
    else:
        raise ValueError("mode must be file or data")


# inference
def classification4type(file, checkpoint='model_checkpoint_utf8_30.pth', batchsize=16, mode='file', result_path=None):
    """
    调用bert-base-chinese模型进行孕前和产后的分类
    :param mode: file or data
    :param data:
    :param result_path: 推理后的数据的保存位置
    :param checkpoint: 选择的模型
    :param batchsize: 推理批量大小，适应不同显存
    :return:
    """
    data = input_check(file)
    df = data.df
    assert data.target_column in df.columns, "the target column not in file"
    inference_data = list(df[data.target_column])
    tokenizer = Tokenizer()
    inference_encodings = tokenizer(inference_data)
    inference_labels = [0 for i in range(len(inference_data))]
    test_dataset = TextClassificationDataset(inference_encodings, labels=inference_labels)

    model = Model()

    checkpoint = os.path.join('./models/checkpoints', checkpoint)
    predictions = inference(test_dataset, model, checkpoint,
                            batchsize, shuffle=False, mode="output")
    df['type'] = pd.Series(predictions, name='type')
    return output_check(df, result_path, mode)


# subtitle
def classification4subtitle(file, subtitle, mode='file', result_path=None):
    data = input_check(file)
    df = data.df
    assert type(subtitle) == Subtitle, "subtitle must be a Subtitle Class"
    result = df[data.target_column].apply(contain_keyword, arg=(subtitle.keywords,))
    df['subtitle'] = pd.Series(result, name='subtitle')
    return output_check(df, result_path, mode)





import io
import os

import chardet
import pandas as pd

from models.model import Model
from models.utils import Tokenizer, TextClassificationDataset
from utils import inference


# 分类分为三阶段
# 一阶段为所处时期的分类(孕前和产后)，为互斥分类，采用深度学习模型
# 二阶段和三阶段为副标题和标签的分类，为非互斥分类，采用关键词提取
# 分类顺序：一阶段->二阶段->三阶段


# inference
def classification4type(filepath, checkpoint='model_checkpoint_utf8_30.pth',
                        column_name='combinedText', batchsize=16, result_path=None):
    """
    调用bert-base-chinese模型进行孕前和产后的分类
    :param filepath: 被推理的数据的位置;或者bytes对象的文件
    :param result_path: 推理后的数据的保存位置
    :param checkpoint: 选择的模型
    :param column_name: 被推理的文本所在csv文件的列名
    :param batchsize: 推理批量大小，适应不同显存
    :return:
    """
    if type(filepath) == str:
        with open(filepath, mode='rb') as f:
            content = f.read()
        result = chardet.detect(content)
        df = pd.read_csv(filepath, encoding="ansi" if result['encoding'][:2] == "GB" else result['encoding'])
    elif type(filepath) == bytes:
        file = filepath
        result = chardet.detect(file)
        file = io.BytesIO(file)
        df = pd.read_csv(file, encoding="ansi" if result['encoding'][:2] == "GB" else result['encoding'])
    else:
        raise TypeError("filepath parameter must be a path of a data table or a bytes object of data table")
    assert column_name in df.columns, "the target column not in file"
    inference_data = list(df[column_name])
    tokenizer = Tokenizer()
    inference_encodings = tokenizer(inference_data)
    inference_labels = [0 for i in range(len(inference_data))]
    test_dataset = TextClassificationDataset(inference_encodings, labels=inference_labels)

    model = Model()

    checkpoint = os.path.join('./models/checkpoints', checkpoint)
    predictions = inference(test_dataset, model, checkpoint,
                            batchsize, shuffle=False, mode="output")
    df['type'] = pd.Series(predictions, name='type')
    if result_path is not None:
        output_path = result_path
        df.to_csv(output_path, encoding='utf-8-sig', index=False)
        return f"Output was saved in {output_path}\n Ctrl + left click to preview."
    else:
        output_path = os.path.join(os.getcwd(), r"./temp_files/output.csv")
        df.to_csv(output_path, encoding='utf-8-sig', index=False)
        return output_path



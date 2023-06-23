import torch
from transformers import BertForSequenceClassification


class Model(torch.nn.Module):
    """
    a model from pre-trained "bert-base-chinese"
    """
    def __init__(self, num_labels=2, model_path="bert-base-chinese"):
        super(Model, self).__init__()
        self.num_labels = num_labels
        # 加载预训练模型
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=self.num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

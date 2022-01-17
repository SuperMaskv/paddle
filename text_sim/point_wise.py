import paddle.nn as nn
import paddle.nn.functional as F


class PointwiseMatching(nn.Layer):
    def __init__(self, pretrained_model, dropout=None):
        super(PointwiseMatching, self).__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(self.ptm.config['hidden_size'], 2)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        cls_embedding = self.dropout(cls_embedding)

        return F.softmax(self.classifier(cls_embedding))

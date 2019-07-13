from torch import nn
from model.encoder import Encoder
import torch.nn.functional as F


class PairwiseBERT(nn.Module):
    def __init__(self, pretrained_version, repr_size):
        super(PairwiseBERT, self).__init__()
        self.pretrained_version = pretrained_version
        self.repr_size = repr_size
        self.q_bert = Encoder.from_pretrained(self.pretrained_version)
        self.p_bert = Encoder.from_pretrained(self.pretrained_version)
        self.q_repr = nn.Linear(self.q_bert.config.hidden_size, self.repr_size)
        self.p_repr = nn.Linear(self.p_bert.config.hidden_size, self.repr_size)

    def forward(self, input_ids, is_query):
        if is_query:
            input_features = self.q_bert(input_ids=input_ids,
                                        output_all_encoded_layers=False,
                                        output_final_multi_head_repr=False)
            return self.q_repr(input_features[:,0])
        else:
            input_features = self.p_bert(input_ids=input_ids,
                                        output_all_encoded_layers=False,
                                        output_final_multi_head_repr=False)
            return self.p_repr(input_features[:,0])


class PointwiseBERT(nn.Module):
    def __init__(self, pretrained_version, repr_size):
        super(PointwiseBERT, self).__init__()
        self.pretrained_version = pretrained_version
        self.repr_size = repr_size
        self.q_bert = Encoder.from_pretrained(self.pretrained_version)
        self.p_bert = Encoder.from_pretrained(self.pretrained_version)
        self.q_repr = nn.Linear(self.q_bert.config.hidden_size, self.repr_size)
        self.p_repr = nn.Linear(self.p_bert.config.hidden_size, self.repr_size)
        self.logits = nn.Linear(self.repr_size*2, 2)

    def forward(self, input_ids, is_query):
        if is_query:
            input_features = self.q_bert(input_ids=input_ids,
                                        output_all_encoded_layers=False,
                                        output_final_multi_head_repr=False)
            return self.q_repr(input_features[:,0])
        else:
            input_features = self.p_bert(input_ids=input_ids,
                                        output_all_encoded_layers=False,
                                        output_final_multi_head_repr=False)
            return self.p_repr(input_features[:,0])

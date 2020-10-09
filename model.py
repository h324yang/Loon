import torch
from torch import nn
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModel
)


ORIGINAL_MODELS = {
    "bert-base-multilingual-cased": True
}


class PairwiseBERT(nn.Module):
    def __init__(self, bert, dim, *model_args, **kwargs):
        super(PairwiseBERT, self).__init__()
        self.bert = bert
        self.dim = dim
        self.fc = nn.Linear(
            self.bert.config.hidden_size, 
            self.dim
        )
        
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        dim=100, 
        *model_args, 
        **kwargs
    ):
        if pretrained_model_name_or_path in ORIGINAL_MODELS:
            bert = AutoModel.from_pretrained(
                    pretrained_model_name_or_path, 
                    *model_args, 
                    **kwargs
            )

            return cls(bert, dim, *model_args, **kwargs)
        
        else:
            return torch.load(pretrained_model_name_or_path)
            

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask, return_dict=True)
        sent_repr = self.fc(bert_output.pooler_output)
        return sent_repr
    
    def save_pretrained(self, fpath):
        torch.save(self, fpath)
    

def compute_loss(model, batch, margin=3.0, norm=1):
    device = next(model.parameters()).device
    repr_center = model(
        input_ids = batch[0].to(device),
        attention_mask = batch[1].to(device)
    )

    repr_pos = model(
        input_ids = batch[2].to(device),
        attention_mask = batch[3].to(device)
    )
    
    repr_neg = model(
        input_ids = batch[4].to(device),
        attention_mask = batch[5].to(device)
    )
    
    dist = nn.PairwiseDistance(p=norm)
    ranking_loss = nn.MarginRankingLoss(margin)
    pos_dist = dist(repr_center, repr_pos)
    neg_dist = dist(repr_center, repr_neg)
    target = torch.ones_like(repr_center[:, 0]).float().to(device)
    loss = ranking_loss(pos_dist, neg_dist, -1*target)
    return loss


def compute_desc_reprs(model, batch):
    device = next(model.parameters()).device
    batch_reprs = {}
    outputs = model(
        input_ids = batch[1].to(device),
        attention_mask = batch[2].to(device)
    )
    for desc_id, desc_repr in zip(batch[0], outputs):
        desc_id = str(int(desc_id.item()))
        batch_reprs[desc_id] = desc_repr.cpu().numpy()
    return batch_reprs
        

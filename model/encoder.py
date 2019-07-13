from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from model.modeling import BertPreTrainedModel, BertModel

import torch.nn as nn

class Encoder(BertPreTrainedModel):
  def __init__(self, config):
    super(Encoder, self).__init__(config)
    self._config = config
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.apply(self.init_bert_weights)

  def forward(self,
              input_ids,
              token_type_ids=None,
              attention_mask=None,
              output_all_encoded_layers=False,
              output_final_multi_head_repr=False,
              selected_non_final_layers=None,
              route_path=None,
              no_dropout=False):
    """

    :param input_ids:
    :param token_type_ids:
    :param attention_mask:
    :param output_all_encoded_layers:
    :param output_final_multi_head_repr:
    :param selected_non_final_layers:
    :param route_path: always None, this is just a placeholder for branching encoder
    :param no_dropout:
    :return:
    """
    sequence_output, _, final_multi_head_repr = self.bert(
      input_ids, token_type_ids, attention_mask,
      output_all_encoded_layers=output_all_encoded_layers,
      output_final_multi_head_repr=output_final_multi_head_repr,
      selected_non_final_layers=selected_non_final_layers)
    if not no_dropout:
      if output_all_encoded_layers or selected_non_final_layers is not None:
        sequence_output = [self.dropout(seq) for seq in sequence_output]
      else:
        sequence_output = self.dropout(sequence_output)
      if output_final_multi_head_repr:
        final_multi_head_repr = self.dropout(final_multi_head_repr)
        return sequence_output, final_multi_head_repr
      else:
        return sequence_output
    else:
      if output_final_multi_head_repr:
        return sequence_output, final_multi_head_repr
      else:
        return sequence_output

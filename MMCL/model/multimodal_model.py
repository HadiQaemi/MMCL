import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel
from torch.nn import LayerNorm
from .transformer_block import TransformerBlock
from .mlp import Mlp

class MultimodalModel(nn.Module):
    def __init__(
              self,
              num_labels,
              intermediate_dim: int = 512,
              pretrained_text_name: str = 'bert-base-uncased',
              pretrained_image_name: str = "microsoft/swin-tiny-patch4-window7-224"
            ):

        super(MultimodalModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name
        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )
        self.transformer_encoder = TransformerBlock(768, intermediate_dim, self.num_labels)
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.Mlp = Mlp(768+768, intermediate_dim, self.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):

        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        encoded_text = self.attention_norm(encoded_text['last_hidden_state'])
        encoded_image = self.attention_norm(encoded_image['last_hidden_state'])
        out1 = self.transformer_encoder(encoded_text, encoded_image, encoded_text)
        out2 = self.transformer_encoder(encoded_image, encoded_text, encoded_image)

        attention_mean_1 = out1.mean(axis=1)
        attention_mean_2 = out2.mean(axis=1)

        attention_mean_1 = self.attention_norm(attention_mean_1)
        attention_mean_2 = self.attention_norm(attention_mean_2)
        logits = self.Mlp(torch.cat(
            [
                attention_mean_1,
                attention_mean_2,
            ],
            dim=1
        ))

        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss

        return out
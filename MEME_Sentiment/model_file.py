import torch
import torch.nn as nn
from transformers import ViTModel, BertModel

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_q, kdim=dim_kv, vdim=dim_kv, num_heads=num_heads, batch_first=True)

    def forward(self, query, key_value):
        out, _ = self.attn(query, key_value, key_value)
        return out

class MemeCrossAttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.cross_attn_img_to_txt = CrossAttentionBlock(dim_q=768, dim_kv=768)
        self.cross_attn_txt_to_img = CrossAttentionBlock(dim_q=768, dim_kv=768)

        self.classifier_input_dim = 768 * 2

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.humor_classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

        self.sarcasm_classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

        self.offense_classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, image, text):
        vit_out = self.vit(pixel_values=image).last_hidden_state
        bert_out = self.bert(**text).last_hidden_state

        img_cls = vit_out[:, 0:1, :]
        text_cls = bert_out[:, 0:1, :]

        img_attn = self.cross_attn_img_to_txt(img_cls, bert_out)
        text_attn = self.cross_attn_txt_to_img(text_cls, vit_out)

        combined = torch.cat([img_attn.squeeze(1), text_attn.squeeze(1)], dim=-1)

        return {
            'sentiment_logits': self.sentiment_classifier(combined),
            'humor_logits': self.humor_classifier(combined),
            'sarcasm_logits': self.sarcasm_classifier(combined),
            'offense_logits': self.offense_classifier(combined),
        }
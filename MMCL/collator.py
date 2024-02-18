# from transformers import AutoTokenizer, AutoFeatureExtractor
from typing import List
import torch
import torch.nn as nn
from PIL import Image
import os

class Collator:
    # tokenizer: AutoTokenizer
    # preprocessor: AutoFeatureExtractor
    def __init__(self, tokenizer, preprocessor, config):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.config = config

    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding='max_length',
            add_special_tokens = True,
            max_length=197,
            # max_length=49,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[Image.open(os.path.join(self.config.path_images, image_id)).convert('RGB') for image_id in images],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }

    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict['text']
                if isinstance(raw_batch_dict, dict) else
                [i['text'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_id']
                if isinstance(raw_batch_dict, dict) else
                [i['image_id'] for i in raw_batch_dict]
            ),
            'labels': torch.tensor(
                raw_batch_dict['label']
                if isinstance(raw_batch_dict, dict) else
                [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }
from transformers import AutoTokenizer, AutoFeatureExtractor
from .collator import Collator
from .model.multimodal_model import MultimodalModel
import torch

def CreatMultimodalCollator(text='bert-base-uncased', image="CreatMultimodalCollator", num_labels=5, config=""):
    tokenizer = AutoTokenizer.from_pretrained(text)
    preprocessor = AutoFeatureExtractor.from_pretrained(image)

    multi_collator = Collator(tokenizer=tokenizer, preprocessor=preprocessor, config=config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    multi_model = MultimodalModel(num_labels=num_labels, pretrained_text_name=text, pretrained_image_name=image).to(device)
    return multi_collator, multi_model
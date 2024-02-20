from transformers import (TrainingArguments, Trainer)
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import numpy as np
import argparse
from MMCL.utils import load_config_as_namespace, load_data
from MMCL.create_multimodal_collator import CreatMultimodalCollator
from sklearn.metrics import accuracy_score, f1_score
import os
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
   
def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_tuple
    preds = logits.argmax(axis=-1)
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    } 

def TrainModel(dataset, args, tags_space, config, text_model='bert-base-uncased', image_model='google/vit-base-patch16-224-in21k', multimodal_model='bert_vit'):
    collator, model = CreatMultimodalCollator(text_model, image_model, len(tags_space), config)

    multi_args = deepcopy(args)
    multi_args.output_dir = os.path.join("..", "checkpoint", multimodal_model)
    multi_trainer = Trainer(
        model,
        multi_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    train_multi_metrics = multi_trainer.train()
    eval_multi_metrics = multi_trainer.evaluate()

    return collator, model, train_multi_metrics, eval_multi_metrics
    

def create_parser():
    parser = argparse.ArgumentParser(description="Multimodal Classifier")
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to the log directory')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the log directory')
    return parser
    
parser = create_parser()
args = parser.parse_args()
config = load_config_as_namespace(args.config)
config.log_dir = args.log_dir
dataset, tags_space = load_data(config)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    config = load_config_as_namespace(args.config)
    config.log_dir = args.log_dir
    args = TrainingArguments(
        output_dir=config.output_dir,
        seed=config.seed,
        eval_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        metric_for_best_model=config.metric_for_best_model,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        remove_unused_columns=config.remove_unused_columns,
        num_train_epochs=config.num_train_epochs,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        load_best_model_at_end=config.load_best_model_at_end,
    )
    collator, model, train_multi_metrics, eval_multi_metrics = TrainModel(dataset, args, tags_space, config=config)
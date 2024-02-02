from datasets import load_dataset

import argparse
import yaml

def load_config_as_namespace(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)

def load_data(config_file):
  dataset = load_dataset(
      "csv",
      data_files={
          "train": config_file.train_dataset,
          "test": config_file.test_dataset
      }
  )

  with open(config_file.tages_files) as f:
      tags_space = f.read().splitlines()

  return dataset.map(
      lambda examples: {
          'label': [
              tags_space.index(ans.replace(" ", "").split(",")[0])
              for ans in examples['label']
          ]
      },
      batched=True
  ), tags_space
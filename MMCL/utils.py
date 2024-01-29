from datasets import load_dataset

def load_data(train_dataset, test_dataset, tages_files):
  dataset = load_dataset(
      "csv",
      data_files={
          "train": train_dataset,
          "test": test_dataset
      }
  )

  with open(tages_files) as f:
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
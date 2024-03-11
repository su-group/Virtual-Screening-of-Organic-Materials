from sklearn.model_selection import train_test_split
import pkg_resources
import torch
from rxnfp.models import SmilesClassificationModel
import pandas as pd

import os

import wandb

data_name = 'MpPD'
data_path = 'data/fine_turning_data/MpDB/MpDB.csv'
df = pd.read_csv(data_path, encoding='utf-8')
labels = 'E_gap'

model_path = f'best_pre_model/pre-{pre_model}'

# tmpls = []
# for index, i in df.iterrows():
#    tmpls.append(i['Chromophore'] + i['Solvent'])
# df['smiles'] = tmpls
df = df[['smiles', labels]]
df = df.dropna(subset=[labels])[['smiles', labels]]
print(f"useful data count: {len(df)}")
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
eval_df, test_df = train_test_split(eval_df, test_size=0.5, random_state=42)

train_file = pd.DataFrame(train_df)
train_file.to_csv(f"{data_path.replace('.csv', '_train.csv')}", encoding='utf-8', index=False)
eval_file = pd.DataFrame(eval_df)
eval_file.to_csv(f"{data_path.replace('.csv', '_val.csv')}", encoding='utf-8', index=False)
test_file = pd.DataFrame(test_df)
test_file.to_csv(f"{data_path.replace('.csv', '_test.csv')}", encoding='utf-8', index=False)

train_df.columns = ['text', 'labels']
eval_df.columns = ['text', 'labels']

mean = train_df['labels'].mean()
std = train_df['labels'].std()
train_df['labels'] = train_df['labels'].apply(lambda x: (x - mean) / std)

eval_df['labels'] = eval_df['labels'].apply(lambda x: (x - mean) / std)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(f"cuda is: {torch.cuda.is_available()}")

model_args = {
    'num_train_epochs': 50, 'overwrite_output_dir': True,
    'learning_rate': 0.00001, 'gradient_accumulation_steps': 1,
    'regression': True, "num_labels": 1, "fp16": False,
    "evaluate_during_training": True, 'manual_seed': 42,
    "max_seq_length": 300, "train_batch_size": 64, "warmup_ratio": 0.00,
    "config": {'hidden_dropout_prob': 0.4},
    'wandb_project': f"FT-{pre_model}-{data_name}-{labels}",
    'best_model_dir': f"best_model/{pre_model}-{data_name}-{labels}",
    'output_dir': f'out/{pre_model}-{data_name}-{labels}',
}


model = SmilesClassificationModel("bert", model_path, num_labels=1,
                                  args=model_args, use_cuda=torch.cuda.is_available())

model.train_model(train_df, eval_df=eval_df)

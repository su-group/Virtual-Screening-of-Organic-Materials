import torch
import pandas as pd
from rxnfp.models import SmilesLanguageModelingModel
import os

from sklearn.model_selection import train_test_split


df = pd.read_csv('data/pretrain_data/USPTO.csv')
data_name = 'USPTO'
df.columns = ['smiles']
train_file = 'data/USPTO/USPTO_train.csv'
eval_file = train_file.replace('train', 'val')
# df = df.sample(1000000, random_state=42, replace=False)
train, test = train_test_split(df, test_size=0.1, random_state=42)
train_files = pd.DataFrame(train['smiles'])
train_files.to_csv(train_file, encoding='utf-8', index=False)

eval_files = pd.DataFrame(test)
eval_files.to_csv(eval_file, encoding='utf-8', index=False)
print(len(train_files), len(eval_files), len(df))

print(f"cuda is: {torch.cuda.is_available()}")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = {
    "architectures": [
        "BertForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 4,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "type_vocab_size": 2,

}

vocab_path = 'data/vocab.txt'

args = {'config': config,
        'vocab_path': vocab_path,
        'train_batch_size': 64,
        'manual_seed': 42,
        'fp16': False,
        "num_train_epochs": 20,
        'max_seq_length': 300,
        'evaluate_during_training': True,
        'overwrite_output_dir': True,
        'output_dir': f'out/pre-{data_name}',
        'learning_rate': 5e-5,
        'wandb_project': f"pretrain_{data_name}",
        'best_model_dir': f"best_pre_model/pre-{data_name}"
        }

model = SmilesLanguageModelingModel(model_type='bert', model_name=None, args=args, use_cuda=torch.cuda.is_available())
# print(model.model)
model.train_model(train_file=train_file, eval_file=eval_file)

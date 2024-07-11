import argparse

import torch
import pandas as pd
from rxnfp.models import SmilesLanguageModelingModel
import os
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pretraining Pipeline for Smiles Language Modeling')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output')
    parser.add_argument('--data_name', type=str, default='USPTO', help='Name of the dataset')
    parser.add_argument('--split_ratio', type=float, default=0.1, help='Ratio for train-test split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()
class PretrainingPipeline:
    def __init__(self, data_path, output_dir, data_name='USPTO', split_ratio=0.1, seed=42):
        self.data_path = data_path
        self.output_dir = output_dir
        self.data_name = data_name
        self.split_ratio = split_ratio
        self.seed = seed
        self.config = self._get_config()
        self.vocab_path = 'data/vocab.txt'
        self.args = self._get_args()

    def _get_config(self):
        return {
            "architectures": ["BertForMaskedLM"],
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
            "type_vocab_size": 2,
        }

    def _get_args(self):
        return {
            'config': self.config,
            'vocab_path': self.vocab_path,
            'train_batch_size': 64,
            'manual_seed': self.seed,
            'fp16': False,
            'num_train_epochs': 20,
            'max_seq_length': 300,
            'evaluate_during_training': True,
            'overwrite_output_dir': True,
            'output_dir': f'{self.output_dir}/pre-{self.data_name}',
            'learning_rate': 5e-5,
            'wandb_project': f"pretrain_{self.data_name}",
            'best_model_dir': f"{self.output_dir}/best_pre_model/pre-{self.data_name}"
        }

    def prepare_data(self):
        df = pd.read_csv(self.data_path)
        df.columns = ['smiles']
        train, test = train_test_split(df, test_size=self.split_ratio, random_state=self.seed)
        train.to_csv(f'{self.output_dir}/{self.data_name}_train.csv', encoding='utf-8', index=False)
        test.to_csv(f'{self.output_dir}/{self.data_name}_val.csv', encoding='utf-8', index=False)
        print(len(train), len(test), len(df))
        return df

    def setup_environment(self):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print(f"cuda is: {torch.cuda.is_available()}")

    def train_model(self):
        model = SmilesLanguageModelingModel(
            model_type='bert',
            model_name=None,
            args=self.args,
            use_cuda=torch.cuda.is_available()
        )
        model.train_model(
            train_file=f'{self.output_dir}/{self.data_name}_train.csv',
            eval_file=f'{self.output_dir}/{self.data_name}_val.csv'
        )

if __name__ == "__main__":
    args = parse_arguments()
    pipeline = PretrainingPipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        data_name=args.data_name,
        split_ratio=args.split_ratio,
        seed=args.seed
    )
    pipeline.setup_environment()
    pipeline.prepare_data()
    pipeline.train_model()
# python pretraining.py --data_path data/pretrain_data/USPTO.csv --output_dir out --data_name USPTO --split_ratio 0.1 --seed 42

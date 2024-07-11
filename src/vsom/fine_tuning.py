import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from rxnfp.models import SmilesClassificationModel

class TrainingPipeline:
    def __init__(self, data_name, data_path, labels, pre_model):
        self.data_name = data_name
        self.data_path = data_path
        self.labels = labels
        self.pre_model = pre_model
        self.model_path = f'best_pre_model/pre-{self.pre_model}'
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_args = {
            'num_train_epochs': 50,
            'overwrite_output_dir': True,
            'learning_rate': 1e-5,
            'gradient_accumulation_steps': 1,
            'regression': True,
            'num_labels': 1,
            'fp16': False,
            'evaluate_during_training': True,
            'manual_seed': self.seed,
            'max_seq_length': 300,
            'train_batch_size': 64,
            'warmup_ratio': 0.00,
            'config': {'hidden_dropout_prob': 0.4},
            'wandb_project': f"FT-{self.pre_model}-{self.data_name}-{self.labels}",
            'best_model_dir': f"best_model/{self.pre_model}-{self.data_name}-{self.labels}",
            'output_dir': f'out/{self.pre_model}-{self.data_name}-{self.labels}',
        }

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path, encoding='utf-8')
        df = df[['smiles', self.labels]].dropna(subset=[self.labels])
        print(f"useful data count: {len(df)}")

        # Split data into train, eval, and test sets
        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=self.seed)
        eval_df, test_df = train_test_split(eval_df, test_size=0.5, random_state=self.seed)

        # Save split datasets
        train_df.to_csv(self.data_path.replace('.csv', '_train.csv'), encoding='utf-8', index=False)
        eval_df.to_csv(self.data_path.replace('.csv', '_val.csv'), encoding='utf-8', index=False)
        test_df.to_csv(self.data_path.replace('.csv', '_test.csv'), encoding='utf-8', index=False)

        # Normalize labels
        mean = train_df[self.labels].mean()
        std = train_df[self.labels].std()
        train_df[self.labels] = (train_df[self.labels] - mean) / std
        eval_df[self.labels] = (eval_df[self.labels] - mean) / std

        return train_df, eval_df

    def train(self):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print(f"cuda is: {torch.cuda.is_available()}")

        train_df, eval_df = self.load_and_preprocess_data()

        model = SmilesClassificationModel("bert", self.model_path, num_labels=1,
                                          args=self.model_args, use_cuda=self.device.type == 'cuda')

        model.train_model(train_df, eval_df=eval_df)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Finetune a pre-trained model')
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--labels', type=str, required=True, help='Column name for labels')
    parser.add_argument('--pre_model', type=str, required=True, help='Pretrained model name')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    pipeline = TrainingPipeline(args.data_name, args.data_path, args.labels, args.pre_model)
    pipeline.train()

# python fine_tuning.py --data_name MpPD --data_path data/fine_turning_data/MpDB/MpDB.csv --labels E_gap --pre_model USPTO-SMILES


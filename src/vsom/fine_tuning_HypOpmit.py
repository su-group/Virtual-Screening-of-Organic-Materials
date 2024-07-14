import argparse
import torch
import pkg_resources
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from rxn_yields.core import SmilesClassificationModel
from rxn_yields.data import generate_buchwald_hartwig_rxns
import sklearn

# Load environment variables
load_dotenv(find_dotenv())

# Ensure WandB is available
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False
    print("Warning: WandB is not available.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fine tuning  Hyperparams Optimization"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/solar/solar.csv",
        help="Path to the data file.",
    )
    parser.add_argument(
        "--pre_model",
        type=str,
        default="USPTO_SMILES_clean",
        help="Name of the pre-trained model.",
    )
    parser.add_argument(
        "--data_name", type=str, default="solar", help="Name of the dataset."
    )
    parser.add_argument(
        "--labels", type=str, default="KS_gap", help="Label column name."
    )
    return parser.parse_args()


def preprocess_data(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Preprocess the dataframe by selecting relevant columns and handling missing values."""
    df = df[["smiles", label_col]].dropna(subset=[label_col])
    return df


def standardize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Standardize labels by subtracting mean and dividing by standard deviation."""
    mean = df[label_col].mean()
    std = df[label_col].std()
    df[label_col] = (df[label_col] - mean) / std
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training, evaluation, and test sets."""
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    eval_df, test_df = train_test_split(eval_df, test_size=0.5, random_state=42)
    return train_df, eval_df, test_df


def configure_sweep() -> dict:
    """Configure the hyperparameter sweep using WandB."""
    return {
        "method": "bayes",
        "metric": {"name": "r2", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform", "min": -6, "max": -4},
            "dropout_rate": {"min": 0.05, "max": 0.8},
        },
    }


def train_model(train_df: pd.DataFrame, eval_df: pd.DataFrame):
    """Train the model using the provided datasets."""
    if not wandb_available:
        raise ValueError("WandB is required to run this function.")

    wandb.init()
    print("HyperParams=>>", wandb.config)

    model_args = {
        "wandb_project": f"{args.pre_model}_{args.data_name}_{args.labels}_hy",
        "num_train_epochs": 20,
        "overwrite_output_dir": True,
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.00,
        "train_batch_size": 16,
        "regression": True,
        "num_labels": 1,
        "fp16": False,
        "evaluate_during_training": True,
        "max_seq_length": 300,
        "config": {
            "hidden_dropout_prob": wandb.config.dropout_rate,
        },
        "learning_rate": wandb.config.learning_rate,
    }

    model_path = pkg_resources.resource_filename(
        "rxnfp", f"models/transformers/bert_pretrained"
    )
    pretrained_bert = SmilesClassificationModel(
        "bert",
        model_path,
        num_labels=1,
        args=model_args,
        use_cuda=torch.cuda.is_available(),
    )
    pretrained_bert.train_model(
        train_df,
        output_dir=f"out/{args.pre_model_name}-{args.data_name}-{args.labels}_hy",
        eval_df=eval_df,
        r2=sklearn.metrics.r2_score,
    )


def main():
    global args
    args = parse_arguments()

    # Read data
    df = pd.read_csv(args.data_path, encoding="utf-8")

    # Preprocess data
    df = preprocess_data(df, args.labels)

    # Split data
    train_df, eval_df, _ = split_data(df)

    # Standardize labels
    train_df = standardize_labels(train_df, args.labels)
    eval_df = standardize_labels(eval_df, args.labels)

    # Configure sweep
    sweep_config = configure_sweep()

    # Start sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=f"out/{args.pre_model_name}-{args.data_name}-{args.labels}_hy",
    )
    wandb.agent(sweep_id, function=lambda: train_model(train_df, eval_df))


if __name__ == "__main__":
    main()

import glob
import os
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from rxnfp.models import SmilesClassificationModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置CUDA环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 设置Seaborn样式
sns.set_style("darkgrid")


def get_database(model_path):
    database_map = {
        "USPTO-SMILES": "USPTO_SMILES",
        "uspto-smiles": "USPTO_SMILES",
        "clea": "USPTO_SMILES_clean",
        "chem": "ChEMBL",
        "CEP": "CEPDB",
        "rxn": "rxnfp",
        "deepchem": "DeepChem-77M",
        "MpDB": "MpDB",
        "solar": "Solar",
        "KS_gap": "Solar",
        "Absorption": "EOO_MAW",
        "MAW": "EOO_MAW",
        "abs": "EOO_MAW",
        "Emission": "EOO_MEW",
        "MEW": "EOO_MEW",
        "emi": "EOO_MEW",
        "OPV": "OPV_BDT",
    }

    pretrain_database = database_map.get(model_path.split("-")[0].lower(), "USPTO")
    fine_tuning_database = database_map.get(
        model_path.split("-")[1].lower() if "-" in model_path else "", ""
    )
    df_test_path = (
        f"data/fine_turning_data/{fine_tuning_database}/{fine_tuning_database}_test.csv"
        if fine_tuning_database
        else None
    )

    return df_test_path, pretrain_database, fine_tuning_database


def safe_read_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def draw_regression(truth, predict, name="test", path=None):
    y_test = truth.values
    y_predict = predict

    if len(y_test) != len(y_predict):
        raise ValueError("Length of truth and predict must be the same.")

    df = pd.DataFrame({"truth": y_test, "predict": y_predict})

    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)

    print(f"max_truth: {np.max(y_test):.2f}; min_truth: {np.min(y_test):.2f}")
    print(f"max_pred: {np.max(y_predict):.2f}; min_pred: {np.min(y_predict):.2f}")
    print(f"mean_absolute_error: {mae:.3f}")
    print(f"rmse: {rmse:.3f}")
    print(f"r2 score: {r2:.3f}")

    plot_scatter(df, mae, rmse, r2, name, path)
    plot_histogram(df, name, path)

    return mae, rmse, r2


def plot_scatter(df, mae, rmse, r2, name, path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="truth", y="predict")
    sns.regplot(data=df, x="truth", y="predict", scatter=False, color="red")
    plt.title(
        f"Scatter Plot of Truth vs Predict\nR2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}"
    )
    plt.xlabel("Truth")
    plt.ylabel("Predict")
    plt.savefig(
        os.path.join(path, f"{name}_scatter.tiff") if path else f"{name}_scatter.tiff"
    )
    plt.close()


def plot_histogram(df, name, path):
    plt.figure(figsize=(10, 6))
    sns.histplot(df["predict"] - df["truth"], kde=True)
    plt.title(f"Histogram of Prediction Errors")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.savefig(
        os.path.join(path, f"{name}_histogram.tiff")
        if path
        else f"{name}_histogram.tiff"
    )
    plt.close()


if __name__ == "__main__":
    result_path = "result/result.csv"
    if not os.path.exists(result_path):
        with open(result_path, "w") as f:
            f.write(
                "model_path,r2,rmse,mae,pre-trained database,fine tuning database\n"
            )

    model_paths = glob.glob("best_model/*")
    for model_path in model_paths:
        print(f"Processing model: {model_path}")

        try:
            fold_path = os.path.join("result", os.path.basename(model_path))
            os.makedirs(fold_path, exist_ok=True)

            model = SmilesClassificationModel(
                "bert", model_path, num_labels=1, use_cuda=False
            )

            test_file_path, pretrain, finetuning = get_database(model_path)
            if not test_file_path:
                print(f"No test file path found for {model_path}. Skipping...")
                continue

            test_file = safe_read_csv(test_file_path)
            if test_file is None:
                print(f"Failed to read test file {test_file_path}. Skipping...")
                continue

            train_file = safe_read_csv(test_file_path.replace("test", "train"))
            if train_file is None:
                print(
                    f"Failed to read train file {test_file_path.replace('test', 'train')}. Skipping..."
                )
                continue

            train_file.columns = ["text", "labels"]
            test_file.columns = ["text", "labels"]

            mean = train_file["labels"].mean()
            std = train_file["labels"].std()

            preds_test = model.predict(test_file["text"].values)[0]
            preds_test = preds_test * std + mean

            true_test = test_file["labels"].values

            df_predict = pd.DataFrame(
                {"smiles": test_file["text"], "pred": preds_test, "true": true_test}
            )
            df_predict.to_csv(
                os.path.join(fold_path, "predict.csv"), encoding="utf-8", index=False
            )

            mae, rmse, r2 = draw_regression(
                true_test, preds_test, os.path.basename(model_path), fold_path
            )

            with open(result_path, "a") as f:
                f.write(f"{model_path},{r2},{rmse},{mae},{pretrain},{finetuning}\n")
        except Exception as e:
            print(f"An error occurred while processing {model_path}: {e}")

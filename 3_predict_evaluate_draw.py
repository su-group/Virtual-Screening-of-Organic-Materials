import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from rxnfp.models import SmilesClassificationModel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import glob

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sns.set_style("darkgrid")


def get_database(model_path):
    pretrain_database = ''
    fine_tuning_database = ''
    df_test_path = None

    # get pre-trained database
    if ('clean' in model_path):
        pretrain_database = 'USPTO_SMILES_clean'
    if ('USPTO-SMILES' in model_path) or ('uspto-smiles' in model_path):
        pretrain_database = 'USPTO_SMILES'
    else:
        pretrain_database = 'USPTO'


    # get fine-tuning database
    if ('MpDB' in model_path) or ('MpBD' in model_path) or ('E_gap' in model_path):
        fine_tuning_database = 'MpDB'
        df_test_path = 'data/fine_turning_data/MpDB/MpDB_test.csv'
    if ('phthalo' in model_path) or ('xlogp' in model_path) or ('PcDB' in model_path):
        fine_tuning_database = 'PcDB'
        df_test_path = 'data/fine_turning_data/PcDB/PcDB_test.csv'
    if ('solar' in model_path) or ('KS_gap' in model_path):
        fine_tuning_database = 'Solar'
        df_test_path = 'data/fine_turning_data/Solar/solar_test.csv'
    if ('Absorption' in model_path):
        fine_tuning_database = 'EOO_MAX'
        df_test_path = 'data/fine_turning_data/EOO/EOO_abs_clean_test.csv'
    if ('Emission' in model_path):
        fine_tuning_database = 'OLED_MaxEmission'
        df_test_path = 'data/fine_turning_data/EOO/EOO_emi_clean_test.csv'

    return df_test_path, pretrain_database, fine_tuning_database


def draw_regre(truth, predict, name="test", path=None):
    y_test = list(truth)
    y_predict = list(predict)

    df = pd.DataFrame(columns=['truth', 'predict'])
    df['truth'] = y_test
    df['predict'] = y_predict

    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)
    print(f"max_truth: {max(y_test):.2f}; min_truth: {min(y_test):.2f}")
    print(f"max_pred: {max(y_predict):.2f}; min_pred: {min(y_predict):.2f}")
    print("mean_absolute_error: ", mean_absolute_error(y_test, y_predict))
    print("rmse:", np.sqrt(mean_squared_error(y_test, y_predict)))
    print("r2 score:", r2_score(y_test, y_predict))

    plt.cla()

    # 绘制分布图
    # 绘制对角线
    line_df = pd.DataFrame(columns=["x", "y"])
    line_df["x"] = np.arange(-100, 100)
    line_df["y"] = np.arange(-100, 100)

    sns.relplot(data=df, x="truth", y="predict", legend=False)
    sns.rugplot(data=df, x="truth", y="predict", legend=False)
    ax = sns.lineplot(data=line_df, x="x", y="y", c="black", alpha=0.4)
    ax.lines[0].set_linestyle("--")

    r2_patch = mpatches.Patch(
        label="$R^2$ = {:.3f}".format(r2),
    )
    rmse_patch = mpatches.Patch(
        label="RMSE = {:.3f}".format(rmse),
    )
    mae_patch = mpatches.Patch(
        label="MAE = {:.3f}".format(mae),
    )
    plt.legend(
        handles=[r2_patch, rmse_patch, mae_patch],
        fontsize=10,
    )
    max_lim = max(y_test + y_predict)
    min_lim = min(y_test + y_predict)
    # max_lim = 4
    # min_lim = 1
    plt.xlim(min_lim * 0.95, max_lim * 1.05)
    plt.ylim(min_lim * 0.95, max_lim * 1.05)
    plt.xlabel("Truth ", fontsize=16)
    plt.ylabel("Predict ", fontsize=16)
    if path is None:
        plt.savefig(f"{name}.tiff", dpi=300)
    else:
        plt.savefig(os.path.join(path, f"{name}.tiff"), dpi=300)

    plt.cla()

    difference = df["predict"] - df["truth"]
    difference = pd.DataFrame(difference, columns=["error"])
    ax = sns.histplot(data=difference, x="error", kde=True, bins=20)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # 确保纵坐标为整数

    plt.xlabel("Error", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    # plt.xlim(-2, 2)
    # plt.ylim(0, 500)
    if path is None:
        plt.savefig(f"{name}_error_distribution.tiff", dpi=300)
    else:
        plt.savefig(os.path.join(path, f"{name}_error_distribution.tiff"), dpi=300)
    return mae, rmse, r2


if __name__ == '__main__':
    with open('result/result.csv', 'wb') as f:
        f.write("model_path,r2,rmse,mae,pre-trained database,fine tuning database".encode('utf-8'))
    list_model = list(glob.glob("best_model/*"))
    for model_index in range(len(list_model)):
        model_path = list_model[model_index]
        print(f"{model_index}/{len(list_model)}    ", model_path)
        # load model
        if not ('pre-' in model_path or 'class' in model_path or 'pretrain' in model_path) and "MpPD" in model_path:
            fold_path = r"result/" + model_path.split("/")[-1]
            folder = os.path.exists(fold_path)
            if not folder:
                os.makedirs(fold_path)
            model = SmilesClassificationModel("bert", model_path, num_labels=1,
                                              use_cuda=False)

            # load data
            test_file_path, pretrain, finetuning = get_database(model_path)
            test_file = pd.read_csv(test_file_path)
            train_file = pd.read_csv(test_file_path.replace('test', 'train'))
            # print(f"{pretrain},{finetuning}")
            train_file.columns = ['text', 'labels']
            test_file.columns = ['text', 'labels']
            mean = train_file['labels'].mean()
            std = train_file['labels'].std()
            preds_test = model.predict(test_file['text'].values, )[0]
            preds_test = preds_test * std + mean
            true_test = test_file['labels'].values

            df_predict = pd.DataFrame()
            df_predict['smiles'] = test_file['text']
            df_predict['pred'] = preds_test
            df_predict['true'] = true_test

            df_predict.to_csv(fold_path + '/predict.csv', encoding='utf-8', index=False, index_label=False)
            mae, rmse, r2 = draw_regre(true_test, preds_test, model_path.split("/")[-1], fold_path)
            with open('result/result.csv', 'ab') as f:
                f.write(f"\n{model_path.split('/')[-1]},{r2},{rmse},{mae},{pretrain},{finetuning}".encode('utf-8'))

# Transfer Learning across Different Chemical Domains: Virtual Screening of Organic Materials with Deep Learning Models Pretrained on Small Molecule and Chemical Reaction Data
## ABSTRACT
Machine learning is becoming a preferred method for the virtual screening of organic materials due to its cost-effectiveness over traditional computationally demanding techniques. However, the scarcity of labeled data for organic materials poses a significant challenge for training advanced machine learning models. This study showcases the potential of utilizing databases of drug-like small molecules and chemical reactions to pretrain the BERT model, enhancing its performance in the virtual screening of organic materials. By fine-tuning the BERT models with data from five virtual screening tasks, the version pretrained with the USPTO-SMILES dataset achieved R2 scores exceeding 0.94 for three tasks and over 0.81 for two others. This performance surpasses that of models pretrained on the small molecule or organic materials databases and outperforms three traditional machine learning models trained directly on virtual screening data. The success of the USPTO-SMILES pretrained BERT model can be attributed to the diverse array of organic building blocks in the USPTO database, offering a broader exploration of the chemical space. The study further suggests that accessing a reaction database with a wider range of reactions than the USPTO could further enhance model performance. Overall, this research validates the feasibility of applying transfer learning across different chemical domains for the efficient virtual screening of organic materials.



## Environment

Use the environment provided by [rxn yields]([rxn4chemistry/rxn_yields: Code complementing our manuscript on the prediction of chemical reaction yields (https://iopscience.iop.org/article/10.1088/2632-2153/abc81d) and data augmentation strategies (https://doi.org/10.26434/chemrxiv.13286741). (github.com)](https://github.com/rxn4chemistry/rxn_yields/tree/master/))

```
`conda create -n yields python=3.6 -y
conda activate yields
conda install -c rdkit rdkit=2020.03.3.0 -y
conda install -c tmap tmap -y`
`git clone https://github.com/rxn4chemistry/rxn_yields.git`
`cd rxn_yields`
`pip install -e .`
```

Recommended  installation under ***Linux***

## Data and Models

**The main code for the project is provided here**
We provide the data used, the trained BERT pre-trained model, and the fine-tuned model after transfer learning. Please go to [here](https://doi.org/10.6084/m9.figshare.24679305.v5) to download

Pre- training and fine-tuning data is placed under the `data` folder

pre-training models are placed under `best_pre_model/` folder

Fine-tuning models are placed under `best_model` folder

```
Directory structure
├─ ├── 1_pre_training.py
├─ ├── 2.1 fine_tuning_HOP.py
├─ ├── 2_fine_tuning.py
├─ ├── 3_predict_evaluate_draw.py
├─ ├── README.md
├─ ├── best_model/
├─ ├── best_pre_model/
│   ├─ ├── pre-CEPDB/
│   ├─ ├── pre-ChEMBL/
│   ├─ ├── pre-USPTO/
│   ├─ ├── pre-USPTO-SMILES/
│   ├─ ├── pre-USPTO-SMILES-clear/
│   ├─ ├── vocab.txt
├─ ├── data/
│   ├─ ├── fine_turning_data/
│   │   ├─ ├── EOO/
│   │   ├─ ├── MpDB/
│   │   ├─ ├── PcDB/
│   │   ├─ └── Solar/
│   ├─ └── pretrain_data/
│   │   ├─ ├── CEPDB/
│   │   ├─ ├── USPTO/
│   │   ├─ └── chembl/
```



## Code
<br />`1_pre_training.py` is used for pre-training models. Pre-training datasets can be replaced by modifying read paths.

`2_fine_tuning.py` is used for transfer learning. Fine-tuning datasets can be replaced by modifying read paths.

`2.1 fine_tuning_HOP.py`is used for Bayesian hyperparameter optimization. Bayesian hyperparameter optimization through the use of [wandb](https://wandb.ai/site)

`3_predict_evaluate_draw.py`is used to evaluate models in batches.





## Cited
@misc{zhang2024transfer,<br />      title={Transfer Learning across Different Chemical Domains: Virtual Screening of Organic Materials with Deep Learning Models Pretrained on Small Molecule and Chemical Reaction Data}, <br />      author={Chengwei Zhang and Yushuang Zhai and Ziyang Gong and Hongliang Duan and Yuan-Bin She and Yun-Fang Yang and An Su},<br />      year={2024},<br />      eprint={2311.18377},<br />      archivePrefix={arXiv},<br />      primaryClass={physics.chem-ph}<br />}

# Transfer Learning across Different Chemical Domains: Virtual Screening of Organic Materials with Deep Learning Models Pretrained on Small Molecule and Chemical Reaction Data
## ABSTRACT
Machine learning is becoming a preferred method for the virtual screening of organic materials due to its cost-effectiveness over traditional computationally demanding techniques. However, the scarcity of labeled data for organic materials poses a significant challenge for training advanced machine learning models. This study showcases the potential of utilizing databases of drug-like small molecules and chemical reactions to pretrain the BERT model, enhancing its performance in the virtual screening of organic materials. By fine-tuning the BERT models with data from five virtual screening tasks, the version pretrained with the USPTO-SMILES dataset achieved R2 scores exceeding 0.94 for three tasks and over 0.81 for two others. This performance surpasses that of models pretrained on the small molecule or organic materials databases and outperforms three traditional machine learning models trained directly on virtual screening data. The success of the USPTO-SMILES pretrained BERT model can be attributed to the diverse array of organic building blocks in the USPTO database, offering a broader exploration of the chemical space. The study further suggests that accessing a reaction database with a wider range of reactions than the USPTO could further enhance model performance. Overall, this research validates the feasibility of applying transfer learning across different chemical domains for the efficient virtual screening of organic materials.



## Installation

Recommended  installation under ***Linux***

It is recommended to install this project using **pip**

```
pip install vsom==0.1.5
```

### **Necessary package**

`numpy`

`pandas`

`matplotlib`

`seaborn`

`rdkit`

`torch<2.0.0`

[rxn yields](https://github.com/rxn4chemistry/rxn_yields.git)



If you are having problems installing the environment, you can also try the following process to configure the environment:

```
conda create -n vsom python=3.6
conda activate vsom
conda install matplotlib
conda install seaborn
conda install -c rdkit rdkit=2020.03.3.0 -y
git clone https://github.com/rxn4chemistry/rxn_yields.git
cd rxn_yields
pip install -e .

git clone https://github.com/su-group/Virtual-Screening-of-Organic-Materials.git
cd Virtual-Screening-of-Organic-Materials
pip install -r requirements.txt
pip install --editable .
```



## Data and Models

**The  code for the project is provided here and full projects including data and code are stored in [Figshare](https://doi.org/10.6084/m9.figshare.24679305)**

We provide the data used, the trained BERT pre-trained model, and the fine-tuned model after transfer learning. Please go to [*Figshare*](https://doi.org/10.6084/m9.figshare.24679305) to download

Pre- training and fine-tuning data is placed under the `data` folder

pre-training models are placed under `best_pre_model/` folder

Fine-tuning models are placed under `best_model` folder



## Getting Started
`pretraining.py` is used for pre-training models. Pre-training datasets can be replaced by modifying read paths.

```
e.g. 
python3 -m vsom.pretraining.py --data_path ../data/pretrain_data/USPTO.csv --out_put_dir ../test
```

`fine_tuning.py` is used for transfer learning. Fine-tuning datasets can be replaced by modifying read paths.

```
e.g. 
python3 -m vsom.fine_tuning.py --data_path ../data/solar/solar.csv --pre_model USPTO --labels KS_gap --data_name Solar
```

`fine_tuning_HypOpmit.py`is used for Bayesian hyperparameter optimization. Bayesian hyperparameter optimization through the use of [wandb](https://wandb.ai/site).

```
e.g. 
python3 -m vsom.fine_tuning_HypOpmit.py --data_path ../data/solar/solar.csv --pre_model USPTO --labels KS_gap --data_name Solar
```

`predict_evaluate_draw.py`is used to evaluate models in batches.



## License

This project is licensed under the MIT License.

## External Archive

This project is also archived on *Figshare* with the following DOI: [*10.6084/m9.figshare.24679305*](https://doi.org/10.6084/m9.figshare.24679305)

## Cited
```
@misc{zhang2024transfer,<br />      
title={Transfer Learning across Different Chemical Domains: Virtual Screening of Organic Materials with Deep Learning Models Pretrained on Small Molecule and Chemical Reaction Data}, <br />      
author={Chengwei Zhang and Yushuang Zhai and Ziyang Gong and Hongliang Duan and Yuan-Bin She and Yun-Fang Yang and An Su},<br /> 
year={2024},<br />      
eprint={2311.18377},<br />      
archivePrefix={arXiv},<br />      
primaryClass={physics.chem-ph}<br />}
```


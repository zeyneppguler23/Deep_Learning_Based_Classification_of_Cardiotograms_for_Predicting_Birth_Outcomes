# Deep Learning Based Classification of Cardiotocograms for Predicting Birth Outcomes

The project explores both binary and multi-class classification settings using fetal heart rate (FHR) and uterine contraction (UC) signals, with a particular focus on reproducing and extending CTG-Net-style workflows.

## Overview

The repository is organized around three main tasks:

- preprocessing raw CTG recordings into model-ready windows
- generating labels from expert annotations and majority voting
- training and evaluating deep learning models across repeated cross-validation experiments

The experimental work is notebook-driven, while reusable logic is placed in Python modules under `src/pipeline`, `src/datasetPrep`, and `src/utils`.

## Data and Labels

The code expects WFDB-formatted CTG records and expert annotation files to be available locally in the repository structure.

## Project Directory Map

```text
Deep_Learning_Based_Classification_of_Cardiotograms_for_Predicting_Birth_Outcomes/
|-- README.md
|-- requirements.txt -- Python dependencies used across notebooks and scripts
|-- src/ -- main project code, data folders, and experiment notebooks
|   |-- cleaned_data/ --
|   |   |-- clean_csv/ -- cleaned CTG records saved as CSV files
|   |   `-- lineplots/ -- external lineplot source from: https://github.com/birth-outcomes/ctg_exploratory
|   |-- data/ -- raw data storage
|   |   `-- raw_dataset/ -- downloaded CTU-UHB CTG dataset from: https://www.physionet.org/content/ctu-uhb-ctgdb/1.0.0/
|   |-- datasetPrep/ -- label preparation and dataset setup notebooks
|   |   `-- labelling.ipynb -- builds majority-vote labels and saves step CSVs
|   |-- ExpertAnnotations/ -- final expert-label and majority-vote files
|   |   |-- CTG_Majority_Vote_Labels_FINAL.csv -- final whole-record binary labels
|   |   |-- ExpertAnnCTU-UHB-CTG_20150203 (1).xls -- original expert annotation spreadsheet
|   |   |-- step1_labels.csv
|   |   |-- step2_labels.csv
|   |   |-- step3_labels.csv
|   |   `-- step4_labels.csv
|   |-- notebooks/ -- main notebook experiments
|   |   |-- 01_raw_vs_preprocessed.ipynb -- compares raw and cleaned signals
|   |   |-- 02_Majority_Voting.ipynb -- explores and checks majority voting
|   |   |-- 03_Custom_Model_Binary_Classification.ipynb --EXPERIMENTAL NOTEBOOK custom binary classification experiments
|   |   |-- 04_signal_transformation.ipynb --  EXPERIMENTAL NOTEBOOK signal transformation trials
|   |   |-- 05_ResNet.ipynb -- EXPERIMENTAL NOTEBOOK ResNet-based model experiments
|   |   |-- **07_Replicating_CTGNet.ipynb** -- CTG-Net notebook
|   |   |-- **08_MS_CTG_NET.ipynb** -- MS-CTG-NET classification notebook
|   |   `-- outputs/ -- notebook-generated results and figures
|   |-- pipeline/ -- shared training and evaluation utilities (for the binary classification notebooks)
|   |   |-- balancedSampler.py -- balances classes before training
|   |   |-- CrossValidator.py -- stratified cross-validation helper
|   |   |-- evaluator.py -- computes AUC and threshold metrics
|   |   |-- ExperimentRunner.py -- runs the full experiment loop
|   |   `-- trainer.py -- compiles and trains Keras models
|   |-- Transfer_Learning/ --Notebooks for the multiclass model
|   |   |-- AdressingClassImbalance.ipynb -- main notebook is duplicated for class-imbalance experiments
|   |   |-- **Hybrid-MS-CNN.ipynb** -- MAIN NOTEBOOK FOR hybrid multi-scale CNN experiments
|   |   |-- Input_evaluation.ipynb --  main notebook is duplicated for input ablation and comparison
|   |   `-- outputs/ -- saved transfer-learning results
|   `-- utils/ -- shared helper code
|       `-- visualisation.py -- plotting helpers for metrics and evaluation
`-- .vscode/
```

## How to Run

This repository is mainly designed to be explored through notebooks.

Typical workflow:

1. create and activate a Python environment
2. install dependencies from `requirements.txt`
3. open the repository in VS Code
4. select the correct Python interpreter for the workspace
5. run notebooks from `src/notebooks/` or `src/Transfer_Learning/`

## Outputs

Generated artifacts are typically written to:

- `src/notebooks/outputs/`
- `src/Transfer_Learning/outputs/`
- `outputs/figures/<model_name>/`
- `outputs/tables/`

## Acknowledgment

This work builds on CTG-based fetal monitoring research, including CTG-Net-style deep learning experiments, and uses expert-annotated cardiotocogram data for model development and comparison.

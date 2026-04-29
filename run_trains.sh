#!/usr/bin/env bash
set +e

python train.py --config configs/exiD0-1.yaml
python train.py --config configs/exiD0-2.yaml
python train.py --config configs/exiD0-3.yaml
python train.py --config configs/exiD0-4.yaml
python train.py --config configs/exiD0-5.yaml   


python train.py --config configs/exiD7-1.yaml
python train.py --config configs/exiD7-2.yaml
python train.py --config configs/exiD7-3.yaml
python train.py --config configs/exiD7-4.yaml
python train.py --config configs/exiD7-5.yaml

python evaluate.py --config configs/exiD0-1.yaml --scenario_labels exiD/c7/scenario_labels.csv

python evaluate.py --config configs/exiD0-2.yaml --scenario_labels exiD/c7/scenario_labels.csv

python evaluate.py --config configs/exiD0-3.yaml --scenario_labels exiD/c7/scenario_labels.csv

python evaluate.py --config configs/exiD0-4.yaml --scenario_labels exiD/c7/scenario_labels.csv

python evaluate.py --config configs/exiD0-5.yaml --scenario_labels exiD/c7/scenario_labels.csv

python evaluate.py --config configs/exiD7-1.yaml --scenario_labels exiD/c7/scenario_labels.csv

python evaluate.py --config configs/exiD7-2.yaml --scenario_labels exiD/c7/scenario_labels.csv

python evaluate.py --config configs/exiD7-3.yaml --scenario_labels exiD/c7/scenario_labels.csv

python evaluate.py --config configs/exiD7-4.yaml --scenario_labels exiD/c7/scenario_labels.csv

python evaluate.py --config configs/exiD7-5.yaml --scenario_labels exiD/c7/scenario_labels.csv

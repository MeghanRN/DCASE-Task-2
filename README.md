# DCASE2025 Task 2 Baseline Autoencoder (With Dev Pre-training & Eval Fine-tuning)

This repository implements **Option B**:  
1. **Pre-train** a single autoencoder on all 7 development machines (first‐shot hyperparameter validation & feature learning).  
2. **Fine-tune** that pretrained model separately on each of the 8 evaluation machines (990+10 normals), then **infer** on their test clips.

---

## 📂 Directory Structure

```text
.
├── data/
│   └── dcase2025t2/
│       ├── dev_data/raw/       # 7 dev machines: train/, supplemental/, test/
│       └── eval_data/raw/      # 8 eval machines: train/, supplemental/, test/
├── outputs/
│   └── task2/
│       └── MeghanKret_Cooper_task2_1/
│           └── <Machine>/     # model.pth, threshold.txt, CSVs
├── pretrained/
│   ├── pretrained_dev.pth     # model after dev pre-training
│   └── pretrained_dev_threshold.txt
├── src/
│   ├── config.yaml            # paths & hyperparameters
│   ├── pretrain_dev.py        # pre-train on development machines
│   ├── finetune_and_infer.py  # fine-tune on eval machines + inference
│   └── utils.py               # common feature & IO helpers
├── reports/
│   └── MeghanKret_Cooper_task2_1.technical_report.pdf
├── meta/
│   └── MeghanKret_Cooper_task2_1.meta.yaml
├── download_task2_data.sh     # download & unzip all datasets
├── 03_summarize_results.sh    # aggregate metrics across machines
├── README.md                  # you are here
└── requirements.txt

Installation
pip install -r requirements.txt


Usage
Download datasets

chmod +x download_task2_data.sh
./download_task2_data.sh

Pre-train on Development
python src/pretrain_dev.py
Saves pretrained/pretrained_dev.pth and pretrained/pretrained_dev_threshold.txt

Fine-tune & Infer on Evaluation
python src/finetune_and_infer.py

For each eval machine, saves under outputs/task2/MeghanKret_Cooper_task2_1/<Machine>/:
model.pth
threshold.txt
anomaly_score_<Machine>_section_00_test.csv
decision_result_<Machine>_section_00_test.csv
**Italian Textual Entailment with BERT**

Fine-tuning a Transformer model for Recognizing Textual Entailment (RTE) in Italian using the e-RTE-3-IT benchmark.

This project adapts the Hugging Face GLUE training pipeline to a 3-class RTE task in Italian and implements a complete workflow: preprocessing, fine-tuning, checkpointing and evaluation.

**Repository Structure**

├── e-rte-3-it                        # Dataset

├── finetuning_run_glue_rte_ita.ipynb # Colab notebook to run preprocessing + training

├── run_glue_no_trainer_italiano.py   # Adapted training script

├── finetuning_RTE_ita.pdf            # Full academic project report (IN ITALIAN)

└── README.md

**Task**

Given a premise and a hypothesis, the model predicts one of three classes:

0 -> Entailment

1 -> Contradiction

2 -> Neutral

The dataset was originally distributed in XML format and converted to CSV (sentence1, sentence2, label) for compatibility with the Hugging Face training pipeline.

**Model**

Pretrained backbone:

_dbmdz/bert-base-italian-xxl-cased_

The original run_glue_no_trainer.py script was adapted to:

Remove unused GLUE task parameters

Save checkpoints at the end of each epoch

Automatically select and reload the best model based on validation accuracy

**Training Setup**

Batch size: 8

Learning rate: 1e-5

Epochs: 10

Max sequence length: 128

Stratified 80/20 train-validation split

Metric: Accuracy

**Results**

Test Accuracy: 0.64375

Baselines for comparison:

Random (3 classes): ~0.33

Majority class baseline: ~0.48

The fine-tuned model significantly outperforms both.

**Tech Stack**

Python

PyTorch

Hugging Face Transformers

Hugging Face Datasets

Accelerate

Pandas

Scikit-learn

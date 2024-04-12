# GitBERT

Automated code completion tools have revolutionized software development workflows, yet effectively automating
GitHub workflows remains a challenge due to their complex and context-dependent nature. GitBERT, my submission for a course project, is a novel approach that leverages bidirectional transformer models to enhance the automation of GitHub workflow completion. Unlike previous approaches relying on single-model architectures like T5, GitBERT harnesses the power of BERT, a bidirectional transformer pre-trained on large-scale datasets containing both YAML and English text. GitBERT’s performance is evaluated across various fine-tuning strategies and compared with existing methods like GH-WCOM. The results from this study demonstrate that GitBERT outperforms GH-WCOM in both the Next Sentence (NS) and Job Completion (JC) tasks, achieving higher accuracy, BLEU scores, and ROUGE scores. Notably, GitBERT achieves an accuracy of 48.17% in the JC task compared to GH-WCOM’s 34.23%, showcasing its superior capability in understanding and completing GitHub workflows. These findings underscore the effectiveness of bidirectional transformer architectures in automating complex software development tasks and highlight GitBERT’s potential to enhance developer productivity in CI/CD pipelines.

<br>

## Results

This table illustrates the performance metrics of various learning-rate schedulers applied to different models in two tasks: Next Sentence (NS) and Job Completion (JC). The learning-rate schedulers include Constant, Polynomial, Inverse square root, and Slanted triangular. Each cell in the table represents the accuracy achieved by the corresponding model and fine-tuning strategy, providing insights into their effectiveness in different scenarios.
| Fine-tuning         | Constant | Polynomial | Inverse Sqrt | Slanted    |
|---------------------|----------|------------|--------------|------------|
| NS-base             | 41.31%   | 39.97%     | 38.17%       | 38.08%     |
| NS-pre-trained      | 41.94%   | 41.04%     | 40.64%       | 40.58%     |
| JC-base             | 47.70%   | 45.17%     | 43.17%       | 43.17%     |
| JC-pre-trained      | 48.18%   | 45.98%     | 45.53%       | 45.55%     |

<br>

The table presents the accuracy, BLEU score, and ROGUE score for different models in the NS and JC tasks. The accuracy represents the percentage of correct predictions, while BLEU and ROGUE scores measure the quality of predictions using natural language processing metrics.
| Model       | BLEU  | ROGUE |
|-------------|-------|-------|
| GitBERT-NS  | 43.49%| 48.35%|
| GitBERT-JC  | 55.20%| 59.54%|

<br>

## Setup and Usage

1. Create a New Conda Environment: Run the following command to create a new conda environment named bert_training_env:
`conda create --name bert_training_env python=3.8`
This command creates a new environment with Python version 3.8. You can change the Python version if needed.
2. Activate the Environment: Once the environment is created, activate it using the following command:
`conda activate bert_training_env`
3. Install Required Libraries: Install the required libraries in the activated environment using pip:
`pip install transformers tensorflow tensorflow-text tensorflow_datasets`
This command installs the TensorFlow libraries, Transformers, and other dependencies needed for running the scripts.
4. Run the Scripts: 
    - Run the `tokenize_pretraining.py` script to tokenize YAML and Github workflows for pretraining.
    - Run the `bert_pretraining.py` script to pretrain a BERT-MLM model with the tokenized pretraining data.
    - Run the `tokenize_finetuning.py` script to tokenize either Next-Sentence or Job-Completion data.
    - Once fine-tuning data is tokenized, you can run the `bert_finetuning.py` script to train the BERT model on a specific fine-tuning task i.e Next-Sentence or Job-Completion.
    - The `nlp_metrics.py` script can be used to evaluate the quality of GitBERT's predictions using BLEU and ROUGE scores.

> Datasets are a courtesy of Antonio Mastropaolo, Fiorella Zampetti, Gabriele Bavota, Massimiliano Di Penta curated for their study (GH-WCOM](https://github.com/antonio-mastropaolo/GH-WCOM/)
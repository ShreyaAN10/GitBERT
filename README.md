### Setup

1. Create a New Conda Environment: Run the following command to create a new conda environment named bert_training_env:
`conda create --name bert_training_env python=3.8`
This command creates a new environment with Python version 3.8. You can change the Python version if needed.
2. Activate the Environment: Once the environment is created, activate it using the following command:
`conda activate bert_training_env`
3. Install Required Libraries: Install the required libraries in the activated environment using pip:
`pip install transformers tensorflow tensorflow-text tensorflow_datasets`
This command installs the TensorFlow libraries, Transformers, and other dependencies needed for running the scripts.
4. Run the Scripts: Now you can run the provided Python scripts (`prepare_data.py` and `train_model.py`) within this environment to prepare data and train the BERT model.
    - Run the `prepare_data.py` script to download the dataset and tokenizer
    - Once the data and tokenizer are downloaded, you can run the `train_model.py` script to train the BERT model
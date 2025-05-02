# TimeMultiformer
````commandline
h5py==3.13.0
numpy==2.2.5
six==1.17.0
torch==2.6.0
iTransformer
python==3.10.16
tsdb==0.7.1
````

# After setting up the environment

# Dataset Preparation：
````commandline
# First, download the original dataset:
bash data_downloading.sh
# Dataset processing:
bash dataset_generating.sh
# Run directly
python main.py
# Change the stage in arguments.py according to your needs
# Hyperparameters are in arguments.py
````

# Downstream Classification Tasks
````commandline
# First, perform imputation
# Set the stage in arguments.py to 'impute' mode
# Run
python main.py
# Adjust the RNN mode, using the imputed results as input to train the RNN and test the saved RNN model
python RNN.py
````

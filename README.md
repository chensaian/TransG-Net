# TransG-Net

1. the code of TransG-Net is in TransGNet.py

2. the process of multimodal dataset production is in data_prep.py

3. for specific training settings, please refer to the paper

4. the data used for training and test are in data.csv
   the data is from pubchem and HMDB,due to the copyright we just list the id of molecules in the file.

# Update

# Dependencies
cuda >= 9.0
cudnn >= 7.0
RDKit == 2020.03.4
torch >= 1.4.0 (please upgrade your torch version in order to reduce the training time)
numpy == 1.19.1
scikit-learn == 0.23.2
tqdm == 4.52.0
Tips: Using code conda install -c conda-forge rdkit can help you install package RDKit quickly.

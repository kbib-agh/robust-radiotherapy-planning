Folder contains code for training and testing generative models of anatomy variations. 
The data is composed of pairs (first_fraction_CT, n-th_fraction_CT), created by train_test_data_split.py script.
Based on the first_fraction_CT (which is a conditional variable), a generative model is aimed to learn to generate plausible anatomical variants, whose probability distribution agrees with the distribution of real anatomical variants (n-th_fraction_CTs, for n in range from 2 to typically 30).
The data is split into train and test parts using per-patient assignment (to prevent data lekeage). Train data is further split into five folds in the same manner. 
The assignment of the pairs to train/test splits and folds is in data_dict.json file.

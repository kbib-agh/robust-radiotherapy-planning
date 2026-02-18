This folder contains sample data for training generative models for anatomy variations and methods for estimating probability distribution of delivered dose spatial maps for the purpose of tobust planning.
The files in the data folder are downsized (due to github limits) samples from the original dataset. The files were converted to nifty from original DICOM files (DICOM CT, RT DOSE RT STRUCT).

Conversion of DICOM files to nifty is done with prepareFractionCT.py script which:
1. for each fraction - converts series of DICOM CTs into a 3D nifty
2. for each fraction - extracts OARs of interests and target from RT STRUCT files and creates corresponding 3D segmentations; as there is no consistent naming convention for target, its name (as coded in RT STRUCT) is read from an external file (prostaty.txt in our case - the file contains specific naming of the target used in our database)
3. converts RT DOSE (which is internally 3D) into 3D nifty dose map - planned dose map is available only for the first (planning) fraction

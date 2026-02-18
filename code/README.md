Folder contains code for training and testing generative models of anatomy variations. 

The data is composed of pairs (first_fraction_CT, n-th_fraction_CT), created by train_test_data_split.py script.

Based on the first_fraction_CT (which is a conditional variable), a generative model is aimed to learn to generate plausible anatomical variants, whose probability distribution agrees with the distribution of real anatomical variants (n-th_fraction_CTs, for n in range from 2 to typically 30).

The data is split into train and test parts using per-patient assignment (to prevent data lekeage). Train data is further split into five folds in the same manner. 

The assignment of the pairs to train/test splits and folds is in data_dict.json file.

There are two approaches to generating anatomical variations.

The first approach (in ddf folder) is based on a direct modelling of dense deformation fields (DDF) transforming moving images into fixed images. To create training data, a deformation field between moving and fixed images is first found using SimpleITK. Then, a generative model (Unet with MCDropout) is trained on this data with first fraction CT at the Unet input and DDF at output. At prediction time MCDropout is activated (by leaving network in train state) and then for a fixed first fraction CT at input, different DDF transforms are generated at output as many times, as predictions are done. These DDFs are then used to wrap the fixed first fraction CT to get plausible anatomical variants. Also, DDFs with corresponding anatomical variants are used to get samples from the estimated probability distribution of 3D dose maps.

The second approach is based on Stable Diffusion. First, VAE is used to get latent representation of CTs. Based on these latent representation Conditional Denoising Diffusion (CDD) is trained with the first fraction CT being the conditional variable. At generation time the first fraction CT is compressed to latent representation which conditiones CDD. The output from CDD is then uncompressed with VAE to get anatomical variant of the first fraction CT. Next, DDF between the first fraction CT and generated variant is calculated and all this data is used to get samples from the estimated probability distribution of 3D dose maps.

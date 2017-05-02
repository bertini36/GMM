# Inference in Gaussian Mixture Model

This repo contains different variational methods to learn a Gaussian Mixture 
Model (GMM) and an Univariate Gaussian (UGM) from data. It also contains 
documentation regarding the algorithm derivations, 2D interpolation scripts,
dimensionality reduction scripts, map visualization scripts and other 
interesting models.


## List of available algorithms

### Python
#### Univariate Gaussian (UGM)
- `inference/python/ugm_cavi.py` Coordinate Ascent Variational Inference (CAVI)
algorithm to learn an UGM.
#### Mixture of Gaussians (GMM)
- `inference/python/gmm_means_cavi.py` Coordinate Ascent Variational Inference (CAVI)
 algorithm to learn a GMM with unknown means but known precisions.
- `inference/python/gmm_cavi.py` Coordinate Ascent Variational Inference (CAVI)
 algorithm to learn a GMM with unknown means and unknown precisions.
- `inference/python/gmm_gavi.py` [DOING] Gradient Ascent Variational Inference (GAVI)
 algorithm to learn a GMM with unknown means and unknown precisions.

### Tensorflow 
#### Univariate Gaussian (UGM)
- `inference/tensorflow/ugm_cavi.py` Coordinate Ascent Variational Inference (CAVI) 
 algorithm to learn an UGM.
- `inference/tensorflow/ugm_cavi_linesearch.py` Coordinate Ascent Variational Inference
 (CAVI) with linesearch algorithm to learn an UGM.
- `inference/tensorflow/ugm_gavi.py` contains a Gradient Ascent Variational Inference 
 (GAVI) algorithm to learn an UGM.
#### Mixture of Gaussians (GMM)
- `inference/tensorflow/gmm_means_cavi.py` Coordinate Ascent Variational Inference 
(CAVI) algorithm to learn a GMM with unknown means but known precisions.
- `tinference/tensorflow/gmm_means_cavi_linesearch.py` Coordinate Ascent Variational
 Inference (CAVI) with linesearch algorithm to learn a GMM with unknown 
 means but known precisions.
- `inference/tensorflow/gmm_means_gavi.py` Gradient Ascent Variational Inference 
(GAVI) algorithm to learn a GMM with unknown means but known precisions.

### Autograd
#### Mixture of Gaussians (GMM)
- `inference/autograd/gmm_means_cavi.py` Coordinate Ascent Variational Inferecne (CAVI)
 algorithm to learn a GMM with unknown means but known precisions.
- `inference/autograd/gmm_means.py` [DOING] Coordinate Ascent Variational Inference (CAVI)
 algorithm to learn a GMM with unknown means and unknown precisions.
 
### Edward
#### Univariate Gaussian (UGM)
- `inference/edward/ugm_bbvi.py` [DOING] Black Box Variational Inference (BBVI)
 algorithm to learn an UGM.
#### Mixture of Gaussians (GMM)
- `inference/edward/gmm_bbvi.py` [DOING] Black Box Variational Inference (BBVI)
 algorithm to learn a GMM with unknown means and unknown precisions.


## Data generation scripts
- `data/synthetic/synthetic_data_generator.py` generates data from a mixture of 
 gaussians with different precision matrixs per each components. 
- `data/synthetic/synthetic_data_generator_means.py` generates data from a mixture
 of gaussians with a given precision matrix for all components. 
 
 
## 2D points interpolation scripts
- `preprocessing/interpolation/nn_interpolation.py` Nearest Neighbors interpolation.
- `preprocessing/interpolation/linear_interpolation.R` Linear interpolation.


## Maps generation scripts
- `preprocessing/maps/map.R`  R map
- `preprocessing/maps/gmap.R` R Google map
 
 
## Dimensionality reduction scripts
- `preprocessing/dimReduction/pca.py` Sklearn Principal Component Analysis.
- `preprocessing/dimReduction/ipca.py` Sklearn Incremental Principal
 Component Analysis.
- `preprocessing/dimReduction/ae.py` Keras autoencoder.
- `preprocessing/dimReduction/ppca.py` Edward Probabilistic Principal
 Component Analysis.
 
 
## Other models

### Python
- `models/dirichlet_categorical.py` Exact inference in a Dirichlet Categorical
 model.
- `models/invgamma_normal.py` Exact inference in a Inverse-Gamma Normal
 model.
- `models/NIW_normal.py` Exact inference in a Normal-Inverse-Wishart Normal
 model.
 
### Tensorflow
- `models/linear_regression_tf.py` Linear regression model optimization 
 with Gradient Descent algorithm.

### Autograd
- `models/linear_regression_ag.py` Linear regression model optimization
 with Gradient Descent algorithm.
 
### Edward
- `models/dirichlet_categorical_edward.py` Black Box Variational Inference
 in a Dirichlet Categorical model.
- `models/invgamma_normal_edward.py` Black Box Variational Inference in a 
 Inverse-Gamma Normal model.
- `models/NW_normal_edward.py` [DOING] Black Box Variational Inference in a
 Normal-Wishart Normal model.


## Documentation
- `docs/gmm_means.pdf` contains the derivation of CAVI algorithm for the case
 of unkown means but known precisions. 


## Installation
- Set environment variables $PROJECT and $WORKON_HOME in `setup/provision.sh`
- Execute `setup/provision.sh`
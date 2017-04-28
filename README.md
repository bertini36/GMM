# Inference in Gaussian Mixture Model

This repo contains different variational methods to learn a Gaussian Mixture 
Model (GMM) and an Univariate Gaussian (UGM) from data. It also contains 
documentation regarding the algorithm derivations, 2D interpolation scripts,
dimensionality reduction scripts, map visualization scripts, ...


## List of available algorithms

### Python inference
#### Univariate Gaussian (UGM)
- `pyInference/ugm_cavi.py` Coordinate Ascent Variational Inference (CAVI)
algorithm to learn an UGM.
#### Mixture of Gaussians (GMM)
- `pyInference/gmm_means_cavi.py` Coordinate Ascent Variational Inference (CAVI)
 algorithm to learn a GMM with unknown means but known precisions.
- `pyInference/gmm_cavi.py` Coordinate Ascent Variational Inference (CAVI)
 algorithm to learn a GMM with unknown means and unknown precisions.
- `pyInference/gmm_gavi.py` [DOING] Gradient Ascent Variational Inference (GAVI)
 algorithm to learn a GMM with unknown means and unknown precisions.

### Tensorflow inference
#### Univariate Gaussian (UGM)
- `tfinference/ugm_cavi.py` Coordinate Ascent Variational Inference (CAVI) 
 algorithm to learn an UGM.
- `tfInference/ugm_cavi_linesearch.py` Coordinate Ascent Variational Inference
 (CAVI) with linesearch algorithm to learn an UGM.
- `tfInference/ugm_gavi.py` contains a Gradient Ascent Variational Inference 
 (GAVI) algorithm to learn an UGM.
#### Mixture of Gaussians (GMM)
- `tfInference/gmm_means_cavi.py` Coordinate Ascent Variational Inference 
(CAVI) algorithm to learn a GMM with unknown means but known precisions.
- `tfInference/gmm_means_cavi_linesearch.py` Coordinate Ascent Variational
 Inference (CAVI) with linesearch algorithm to learn a GMM with unknown 
 means but known precisions.
- `tfInference/gmm_means_gavi.py` Gradient Ascent Variational Inference 
(GAVI) algorithm to learn a GMM with unknown means but known precisions.

### Autograd inference
#### Mixture of Gaussians (GMM)
- `agInference/gmm_means_cavi.py` Coordinate Ascent Variational Inferecne (CAVI)
 algorithm to learn a GMM with unknown means but known precisions.
- `agInference/gmm_means.py` [DOING] Coordinate Ascent Variational Inference (CAVI)
 algorithm to learn a GMM with unknown means and unknown precisions.
 
### Edward inference
#### Univariate Gaussian (UGM)
- `edwardInference/ugm_bbvi.py` [DOING] Black Box Variational Inference (BBVI)
 algorithm to learn an UGM.
#### Mixture of Gaussians (GMM)
- `edwardInference/gmm_bbvi.py` [DOING] Black Box Variational Inference (BBVI)
 algorithm to learn a GMM with unknown means and unknown precisions.


## Data generation scripts
- `data/synthetic_data_generator.py` generates data from a mixture of 
 gaussians with different precision matrixs per each components. 
- `data/synthetic_data_generator_means.py` generates data from a mixture
 of gaussians with a given precision matrix for all components. 
 
 
## 2D points interpolation scripts
- `data/real/interpolation/nn_interpolation.py` Nearest Neighbors interpolation.
- `data/real/interpolation/linear_interpolation.R` Linear interpolation.


## Maps generation scripts
- `data/real/maps/map.R`  R map
- `data/real/maps/gmap.R` R Google map
 
 
## Dimensionality reduction scripts
- `data/real/dimensionalityReduction/pca.py` Sklearn Principal Component Analysis.
- `data/real/dimensionalityReduction/ipca.py` Sklearn Incremental Principal
 Component Analysis.
- `data/real/dimensionalityReduction/ae.py` Keras autoencoder.
- `data/real/dimensionalityReduction/ppca.py` Edward Probabilistic Principal
 Component Analysis.


## Documentation
- `docs/gmm_means.pdf` contains the derivation of CAVI algorithm for the case
 of unkown means but known precisions. 


## Installation
- Set environment variables $PROJECT and $WORKON_HOME in `setup/provision.sh`
- Execute `setup/provision.sh`
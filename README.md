# Inference in Gaussian Mixture Model

This repo contains different inference algorithms to learn a Gaussian Mixture 
Model (GMM) from data. It also contains documentation regarding the algorithm 
derivations and scripts to synthetize data. 

## List of available algorithms:

### Python inference
#### Univariate Gaussian (UGM)
1. `pyInference/univariateGaussian/ugm_cavi.py` contains a Coordinate Ascent
Variational Inference (CAVI) algorithm to learn an univariate gaussian .
#### Mixture of Gaussians (GMM)
1. `pyInference/mixtureOfGaussians/knownPrecisions/gmm_means_cavi.py` contains
 a Coordinate Ascent Variational Inference (CAVI) algorithm to learn a GMM
 with unknown means but known precisions.
2. `pyInference/mixtureOfGaussians/complete/gmm_cavi_cavi.py` contains a Coordinate 
 Ascent Variational Inference (CAVI) algorithm to learn a GMM with unknown
 means and unknown precisions.

### Tensorflow inference
#### Univariate Gaussian (UGM)
1. `tfinference/univariateGaussian/ugm_cavi.py` contains a Coordinate Ascent
Variational Inference (CAVI) algorithm to learn an univariate gaussian .
2. `tfInference/univariateGaussian/ugm_cavi_linesearch.py` contains
 a Coordinate Ascent Variational Inference (CAVI) with linesearch algorithm
 to learn an univariate gaussian.
3. `tfInference/univariateGaussian/ugm_gavi.py` contains a Gradient
 Ascent Variational Inference (GAVI) algorithm to learn an univariate gaussian.
#### Mixture of Gaussians (GMM)
1. `tfInference/mixtureGaussians/knownPrecisions/gmm_means_cavi.py`
 contains a Coordinate Ascent Variational Inference (CAVI) algorithm to learn 
 a GMM with unknown means but known precisions.
2. `tfInference/mixtureGaussians/knownPrecisions/gmm_means_cavi_linesearch.py`
 contains a Coordinate Ascent Variational Inference (CAVI) with linesearch 
 algorithm to learn a GMM with unknown means but known precisions.
3. `tfInference/mixtureGaussians/knownPrecisions/gmm_means_gavi.py` 
 contains a Gradient Ascent Variational Inference (GAVI) algorithm to learn a
 GMM with unknown means but known precisions.

## Scripts to generate  data:
1. `data/synthetic_data_generator.py` generates data from a mixture of 
 gaussians with different precision matrixs per each components. 
2. `data/synthetic_data_generator_means.py` generates data from a mixture
 of gaussians with a given precision matrix for all components. 

## Documentation:
1. `docs/gmm_means.pdf` contains the derivation of CAVI algorithm for the case
 of unkown means but known precisions. 

##Installation:
1. Copy `bin/start-env.sh.in` to `bin/start-env.sh`
2. Edit `bin/start-env.sh` to adjust variables such as the project directory and virtualenvwrapper
3. Execute `source bin/provision.sh` 
4. Test by executing `python pyInference/coordAsc/gmm_means.py -K 4 -alpha 1. 1. 1. 1.`
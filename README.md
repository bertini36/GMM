# Inference in Gaussian Mixture Models

This repo contains different inference algorithms to learn a Gaussian Mixture Model (GMM) from data. It also contains documentation regarding the algorithm derications and scripts to synthetize data. 

## List of available algorithms:
1. `pyInference/coordAsc/gmm.py` contains a Coordinate Ascent Variational Inference (CAVI) algorithm implemented in python to learn a GMM with unknown means and Precisions. 
2. `pyInference/coordAsc/gmm.py` contains a CAVI algorithm implemented in python to learn a GMM with unknown means but known Precisions.

## List of scripts to generate  data:
1. `syntheticData.py` generates data from a mixture of gaussians with different precision matrixs per each components. 
2. `syntheticData_means.py`generates data from a mixture of gaussians with a given precision matrix for all components. 

## Documentation:
1. `docs/gmm_means.pdf` contains the derivation of CAVI algorithm for the case of unkown means but known precisions. 

##Installation:
1. Copy `bin/start-env.sh.in` to `bin/start-env.sh`
2. Edit `bin/start-env.sh` to adjust variables such as the project directory and virtualenvwrapper
3. Execute `source bin/provision.sh` 
4. Test by executing `python pyInference/coordAsc/gmm_means.py -K 4 -alpha 1. 1. 1. 1.`
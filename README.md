<h3 align="center">
    bertini36/GMM 📈
</h3>
<h3 align="center">
    Variational Inference in Gaussian Mixture Models
</h3>
<p align="center">
  <a href="https://github.com/bertini36/GMM/blob/master/setup/provision.sh" target="_blank">
    Installation
  </a>&nbsp;&nbsp;•&nbsp;
  <a href="https://github.com/bertini36/GMM/blob/master/inference/" target="_blank">
    Inference strategies
  </a>&nbsp;&nbsp;•&nbsp;
  <a href="https://github.com/bertini36/GMM/blob/master/models/" target="_blank">
    Other models
  </a>&nbsp;&nbsp;•&nbsp;
  <a href="https://github.com/bertini36/GMM/blob/master/docs/doc.pdf" target="_blank">
    Docs
  </a>
</p>
<p align="center">
Variational methods to learn a Gaussian Mixture Model and an Univariate Gaussian from data
</p>
<p align="center">
Powered by <a href="https://github.com/tensorflow/tensorflow" target="_blank">#tensorflow</a>,
<a href="https://github.com/blei-lab/edward" target="_blank">#edward</a>,
 <a href="https://github.com/scipy/scipy" target="_blank">#scipy</a> and
 <a href="https://www.python.org/" target="_blank">#python</a>.
</p>

### 🎯 Inference strategies

#### Univariate Gaussian (UGM)
<a href="https://github.com/bertini36/GMM/blob/master/inference/python/ugm_cavi.py" target="_blank">
    <strong>Python</strong> | Coordinate Ascent Variational Inference
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/tensorflow/ugm_cavi.py" target="_blank">
    <strong>Tensorflow</strong> | Coordinate Ascent Variational Inference 
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/tensorflow/ugm_cavi_linesearch.py" target="_blank">
    <strong>Tensorflow</strong> | Coordinate Ascent Variational Inference with linesearch
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/tensorflow/ugm_gavi.py" target="_blank">
    <strong>Tensorflow</strong> | Gradient Ascent Variational Inference
</a><br>
[BLOCKED] 🚧<a href="https://github.com/bertini36/GMM/blob/master/inference/edward/ugm_bbvi.py" target="_blank">
    <strong>Edward</strong> | Black Box Variational Inference
</a>

 
#### Mixture of Gaussians (GMM)
<a href="https://github.com/bertini36/GMM/blob/master/inference/python/gmm_means_cavi.py" target="_blank">
    <strong>Python</strong> | Coordinate Ascent Variational Inference (unknown means but known precisions)
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/python/gmm_cavi.py" target="_blank">
    <strong>Python</strong> | Coordinate Ascent Variational Inference (unknown and unknown precisions)
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/python/gmm_scavi.py" target="_blank">
    <strong>Python</strong> | Sthocastic Coordinate Ascent Variational Inference (unknown means and unknown precisions)
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/tensorflow/gmm_means_cavi.py" target="_blank">
    <strong>Tensorflow</strong> | Coordinate Ascent Variational Inference (unknown means but known precisions)
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/tensorflow/gmm_means_cavi_linesearch.py" target="_blank">
    <strong>Tensorflow</strong> | Coordinate Ascent Variational Inference with linesearch (unknown means but known precisions)
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/tensorflow/gmm_means_gavi.py" target="_blank">
    <strong>Tensorflow</strong> | Gradient Ascent Variational Inference (unknown means but known precisions)
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/tensorflow/gmm_gavi.py" target="_blank">
    <strong>Tensorflow</strong> | Gradient Ascent Variational Inference (unknown means and unknown precisions)
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/tensorflow/gmm_sgavi.py" target="_blank">
    <strong>Tensorflow</strong> | Sthocastic Gradient Ascent Variational Inference (unknown means and unknown precisions)
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/inference/autograd/gmm_means_cavi.py" target="_blank">
    <strong>Autograd</strong> | Coordinate Ascent Variational Inference (unknown means but known precisions)
</a><br>
[BLOCKED] 🚧<a href="https://github.com/bertini36/GMM/blob/master/inference/autograd/gmm_means.py" target="_blank">
    <strong>Autograd</strong> | Coordinate Ascent Variational Inference (unknown means and unknown precisions)
</a><br>
[BLOCKED] 🚧<a href="https://github.com/bertini36/GMM/blob/master/inference/edward/gmm_bbvi.py" target="_blank">
    <strong>Edward</strong> |  Black Box Variational Inference (unknown means and unknown precisions)
</a>

### 🕺 Other models

<a href="https://github.com/bertini36/GMM/blob/master/models/dirichlet_categorical.py" target="_blank">
    <strong>Python</strong> | Exact inference in a Dirichlet Categorical model
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/models/invgamma_normal.py" target="_blank">
    <strong>Python</strong> | Exact inference in a Inverse-Gamma Normal model
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/models/NIW_normal.py" target="_blank">
    <strong>Python</strong> | Exact inference in a Normal-Inverse-Wishart Normal model
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/models/linear_regression_tf.py" target="_blank">
    <strong>Tensorflow</strong> | Linear regression model optimization with Gradient Descent algorithm
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/models/linear_regression_ag.py" target="_blank">
    <strong>Autograd</strong> | Linear regression model optimization with Gradient Descent algorithm
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/models/dirichlet_categorical_edward.py" target="_blank">
    <strong>Edward</strong> | Black Box Variational Inference in a Dirichlet Categorical model
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/models/invgamma_normal_edward.py" target="_blank">
    <strong>Edward</strong> | Black Box Variational Inference in a Inverse-Gamma Normal model
</a><br>
[BLOCKED] 🚧<a href="https://github.com/bertini36/GMM/blob/master/models/NW_normal_edward.py" target="_blank">
    <strong>Edward</strong> | Black Box Variational Inference in a Normal-Wishart Normal model
</a>
 
### ⛏️ Other scripts

#### Dimensionality reduction scripts
<a href="https://github.com/bertini36/GMM/blob/master/preprocessing/dimReduction/pca.py" target="_blank">
    <strong>Sklearn</strong> | Sklearn Principal Component Analysis
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/preprocessing/dimReduction/ipca.p" target="_blank">
    <strong>Sklearn</strong> | Sklearn Incremental Principal Component Analysis
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/preprocessing/dimReduction/ae.p" target="_blank">
    <strong>Keras</strong> | Autoencoder
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/preprocessing/dimReduction/ppca.p" target="_blank">
    <strong>Edward</strong> | Probabilistic Principal Component Analysis
</a>

#### Data generation scripts
<a href="https://github.com/bertini36/GMM/blob/master/data/synthetic/synthetic_data_generator_means.py" target="_blank">
    <strong>Python</strong> | Mixture of gaussians data generator with different precision matrix per each component
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/data/synthetic/synthetic_data_generator.py" target="_blank">
    <strong>Python</strong> | Mixture of gaussians data generator with a given precision matrix for all components
</a>
 
#### 2D points interpolation scripts
<a href="https://github.com/bertini36/GMM/blob/master/preprocessing/interpolation/nn_interpolation.py" target="_blank">
    <strong>Python</strong> | Nearest Neighbors interpolation
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/preprocessing/interpolation/linear_interpolation.R" target="_blank">
    <strong>R</strong> | Linear interpolation
</a><br>

#### Maps generation scripts
<a href="https://github.com/bertini36/GMM/blob/master/preprocessing/maps/map.R" target="_blank">
    <strong>R</strong> | R map
</a><br>
<a href="https://github.com/bertini36/GMM/blob/master/preprocessing/maps/gmap.R" target="_blank">
    <strong>R</strong> | R Google map
</a><br>

<p align="center">&mdash; Built with :heart: from Barcelona &mdash;</p>
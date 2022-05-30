# Metrizing Fairness

## Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Training](#training)


## Introduction
We study supervised learning problems for predicting properties of individuals who belong to one of two demographic groups, and we seek predictors that are fair according to statistical parity. This means that the distributions of the predictions within the two groups should be close with respect to the Kolmogorov distance, and fairness is achieved by penalizing the dissimilarity of these two distributions in the objective function of the learning problem. In this paper, we showcase conceptual and computational benefits of measuring unfairness with integral probability metrics (IPMs) other than the Kolmogorov distance. Conceptually, we show that the generator of any IPM can be interpreted as a family of utility functions and that unfairness with respect to this IPM arises if individuals in the two demographic groups have diverging expected utilities. We also prove that the unfairness-regularized prediction loss admits unbiased gradient estimators if unfairness is measured by the squared L2-distance or by a squared maximum mean discrepancy. In this case the fair learning problem is susceptible to efficient stochastic gradient descent algorithms. Numerical experiments on real data show that the proposed method outperforms state-of-the-art algorithms for fair learning in that it achieves superior accuracy-unfairness trade-offs---sometimes orders of magnitude faster. Finally, we identify conditions under which statistical parity can improve prediction accuracy.


## Quick Start
The codes in this repository are implemented in Python 3.
To clone the repository:
```
git clone http://github.com//RAO-EPFL/Metrizing-Fairness
cd Metrizing-Fairness
```

To install requirements:
```
pip install -r requirements.txt
```
DCCP package is required for one of the methods that we compare our classifier.
To install the most updated DCCP, please see
```
https://github.com/cvxgrp/dccp/
```
## Training
To train the models in the paper, run this command:
```
python ./src/run.py
```









# Metrizing Fairness

## Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Training](#training)


## Introduction
 We study supervised learning problems that have significant effects on individuals from two demographic groups, and we seek predictors that are fair with respect to a group fairness criterion such as statistical parity (SP). A predictor is SP-fair if the distributions of predictions within the two groups are close in Kolmogorov distance, and fairness is achieved by penalizing the dissimilarity of these two distributions in the objective function of the learning problem. In this paper, we showcase conceptual and computational benefits of measuring unfairness with integral probability metrics (IPMs) other than the Kolmogorov distance. Conceptually, we show that the generator of any IPM can be interpreted as a family of utility functions and that unfairness with respect to this IPM arises if individuals in the two demographic groups have diverging expected utilities. We also prove that the unfairness-regularized prediction loss admits unbiased gradient estimators, which are constructed from random mini-batches of training samples, if unfairness is measured by the squared L2-distance or by a squared maximum mean discrepancy. In this case, the fair learning problem is susceptible to efficient stochastic gradient descent (SGD) algorithms. Numerical experiments on synthetic and real data show that these SGD algorithms outperform state-of-the-art methods for fair learning in that they achieve superior accuracy-unfairness trade-offs---sometimes orders of magnitude faster. Finally, we identify conditions under which unfairness penalties can improve prediction accuracy.


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
DCCP package is required for one of the methods that we compare our classifier to.
To install the most recent DCCP, please see
```
https://github.com/cvxgrp/dccp/
```
## Training
We provide the codes for all experiemnts and baselines used:
- Online Regression (Section 5.1.1): `online_regression` folder
- Online Classification (Section 5.1.2): `online_classification` folder
- Offline Learning (Sections 5.2.1 and 5.2.2): `offline_experiments` folder
- Equal Opportunity (Section 5.2.3): `equal_opportunity` folder










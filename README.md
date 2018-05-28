# AOL
This package provide matlab code to realized 3 popular machine learning method for Dynamic Treatment Regime.
A intermediate version of the paper is https://arxiv.org/abs/1611.02314. 
The paper started in 2012, won a student paper award in 2014 from ASA mental health session. It is currently in press at Stat in Medicine.
We also provide R code for functions in the package in 'DTRlearn'. The matlab code performs more stable than R. 
Please refer to the document of the R package for the inputs and outputs for each function. I provide limited help file now, will add to it latter.

## Dependency
It uses the following package in lasso regression
http://web.stanford.edu/~hastie/glmnet_matlab/

## Selected Functions
### wsvm3
solves the problem of optimizing value function by a weighted SVM  through quadratic programing.

### OLearning_Single
cross validation single stage owl returns the best cost coeff and best treatment for training data.
This is the original Olearning in Zhao 2012

### Olearning_Singlelasso
This is our proposed revised Olearning.
It first takes the residual, and the optimization can also incoorperate negative weights directly.

### Plearningbestlasso
The proposed method in our AOL paper. It solves a multiple stage decision problem through back propogation. In the lense of reinforcement learning, 
we have the most flexible assumption not assuming markov property across stages, the history of last stage were carried over to the current stage.
Each stage is solved by policy based method, and we also borrows the strength of model based learning, to impute 
the missing conterfactural information that appears in multiple stage problem

### Qlearning_Singlelasso
The batch version of Qlearning. It is indeed just fitting a lasso  regression with interaction term

### Qlearning_Dynamiclasso
Multiple stage Qlearning through dynamic programming

### Olearning_Dynamic
Multiple stage Olearning in Zhao 2015

# Reference
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3636816/

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4517946/

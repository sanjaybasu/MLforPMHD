# Machine learning for precision medicine and health disparities, 2018
# Sanjay Basu, basus@stanford.edu

# this code borrows heavily from other sources who have open-sourced their code and deserve credit as primary sources, specifically:
# https://github.com/ledell/sldm4-h2o/blob/master/sldm4-deeplearning-h2o.Rmd
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html
# https://gist.github.com/ledell/102a24eef72bdc4ea2cd876492095684

# first, install and initialize the R packages used in this example [must be connected to the Internet]
rm(list=ls())
pkgs <- c("RCurl","jsonlite","DMwR","caret","lattice","ggplot2","grid","data.table")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-wright/2/R")
# you only have to do the above installation steps once on any given machine

library(h2o)
h2o.init()

# let's create a dataset to analyze, including a variety of different types of variables
options(warn=-1)
pop = 1e5
dummvars = 40 # binary dummy variables, e.g., yes/no for specific covariate values
secovars = 30 # secondary dummy variables, whose existence is influenced by the above-noted dummy variables, e.g., more likely to have a diagnosis if having other co-morbid conditions
catevars = 20 # categorical variables, values 1 through 5, e.g., income category, education, race/ethnicity, etc.
contvars = 10 # continuous variables, e.g., biomarker values
set.seed(100)
dummvarsmeans = runif(dummvars)/3 # producing dummy variables
set.seed(200)
x1=matrix(rbinom(pop*dummvars,1,dummvarsmeans),pop,dummvars, byrow=T) # putting dummy variables into matrix form
secovarselect = round(min(dummvars/2,10))
x2 = matrix(0,pop,secovars) # creating space for secondary dummy variables
set.seed(300)
x2[,1] = rbinom(pop,1,(x1[,1])/secovarselect)
for (i in 2:secovars){
  set.seed(400+i)
  x2[,i]=rbinom(pop,1,rowMeans(x1[,1:i])/secovarselect+runif(pop)/10) # producing secondary dummy variables with some noise to avoid perfect linear predictability
}
set.seed(500)
catevarsshape = round(10*runif(catevars)) # producing categorical variables
set.seed(600)
x3=matrix(round(5*rbeta(pop*catevars, 2, catevarsshape)),pop,catevars, byrow=T) # putting categorical variables into matrix form
set.seed(700)
contvarsshape = round(10*runif(contvars)) # producing con't variables by choosing shape parameters for beta distribution
set.seed(800)
contvarsmag = round(1000*runif(contvars)) # producing con't variables of various magnitudes to reflect claims' data variation in con't variable absolute numbers
set.seed(900)
x4=matrix(round(contvarsmag*rbeta(pop*contvars, 2, contvarsshape)),pop,contvars, byrow=T) # putting con't variables into matrix form
covars = cbind(x1,x2,x3,x4)
dim = dummvars+catevars+contvars+secovars
influencers = seq(1,dim,round(dim/10)) # choosing which variables are truly influential on the primary outcome
set.seed(1000)
coefs = runif(length(influencers))
set.seed(1100)
# simulating probability of the primary outcome based on influential variables, plus some complex interactions among other variables, producing positive and negative heterogeneity, and adding some noise
outprob = (rowMeans(coefs*covars[,influencers])/dim+((covars[,97]*covars[,98]*covars[,99]*covars[,100])/(3*max(covars[,97]*covars[,98]*covars[,99]*covars[,100])))+covars[,round(3*dim/20)]*covars[,round(4*dim/20)]/2-covars[,round(5*dim/20)]*covars[,round(7*dim/20)]/2+rnorm(pop, mean =0, sd = .1)) 
outprob[outprob>1]=1
outprob[outprob<0]=0
set.seed(1200)
y = rbinom(pop,1,outprob) # making the primary outcome variable binary and probabilistic
alldata = data.frame(cbind(x1,x2,x3,x4,y))
colnames(alldata) = c(paste("X",c(1:dim),sep=""),"y")
alldata$y = factor(alldata$y)
library(caret)
set.seed(1300)
splitIndex <- createDataPartition(alldata$y, p = .5, list = FALSE, times = 1) # randomly splitting the data into train and test sets
trainSplit <- alldata[ splitIndex,]
testSplit <- alldata[-splitIndex,]
prop.table(table(trainSplit$y))
library(DMwR)
set.seed(1400)
trainSplit <- SMOTE(y ~ ., trainSplit, perc.over = 100, perc.under=200) # making a balanced subset of the training data, which is known to improve ML methods' performance
prop.table(table(trainSplit$y))
prop.table(table(testSplit$y))

# split up the data into training and testing subsets
train <- as.h2o(trainSplit)
test <- as.h2o(testSplit)
y <- "y"
x <- setdiff(names(train), y)
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])

# let's train a standard logistic regression model as a referent
# let's do elastic net regularization and find the optimal values of penalty parameters in elastic net to minimize our error
# To get the best possible model, we need to find the optimal values of the regularization parameters α and λ. To find the optimal values, H2O allows you to perform a grid search over α and a special form of grid search called “lambda search” over λ
# The recommended way to find optimal regularization settings on H2O is to do a grid search over a few α values with an automatic lambda search for each α
# The alpha parameter controls the distribution between the ℓ1 (LASSO) and ℓ2 (ridge regression) penalties. A value of 1.0 for alpha represents LASSO, and an alpha value of 0.0 produces ridge regression.
# The lambda parameter controls the amount of regularization applied. If lambda is 0.0, no regularization is applied, and the alpha parameter is ignored. The default value for lambda is calculated by H2O using a heuristic based on the training data. If you allow H2O to calculate the value for lambda, you can see the chosen value in the model output.

grid_id <- 'glm_grid'
hyper_parameters <- list( alpha = c(0, .5, 1) )
model_glm_grid <- h2o.grid(
  algorithm = "glm", 
  grid_id = grid_id,
  hyper_params = hyper_parameters,
  training_frame = train, 
  validation_frame = test, 
  x = x, 
  y = y,
  lambda_search = TRUE,
  family = "binomial"
)

# sort the model by the specified evaluation metric
# and obtain the top one (the best model)
stopping_metric <- 'AUC'
sorted_models <- h2o.getGrid(
  grid_id = grid_id, 
  sort_by = stopping_metric,
  decreasing = TRUE
)
best_model <- h2o.getModel(sorted_models@model_ids[[1]])

# for binomial output, h2o will choose the cutoff threshold by 
# maximizing the f1 score by default
h2o.confusionMatrix(best_model, valid = TRUE)

# coefficients (standardized and non-standardized)
# or we can use the short-cut below
# h2o.coef(best_model)
# h2o.coef_norm(best_model)
best_model@model$coefficients

# obtain the regularization, alpha and lambda 
best_model@model$model_summary$regularization

# area under the curve, C-stat
auc <- h2o.auc(best_model, valid = TRUE)
fpr <- h2o.fpr( h2o.performance(best_model, valid = TRUE) )[['fpr']]
tpr <- h2o.tpr( h2o.performance(best_model, valid = TRUE) )[['tpr']]
library(data.table)
ggplot(data.table(fpr = fpr, tpr = tpr), aes(fpr, tpr)) + # ROC curve with false positive rate on x and true positive rate on y
  geom_line() + theme_bw() + ggtitle( sprintf('AUC: %f', auc)) # AUC or C stat

# variable importance plot
h2o.varimp_plot(best_model)

# partial dependence plot
h2o.partialPlot(best_model,data=test,cols=c("X91","X100"))
  

# next, let's try a stacked ensemble of GBM and RF learners


# Number of cross-validation folds (to generate level-one data for stacking)
nfolds <- 5

# There are a few ways to assemble a list of models to stack together:
# 1. Train individual models and put them in a list
# 2. Train a grid of models
# 3. Train several grids of models
# Note: All base models must have the same cross-validation folds and
# the cross-validated predicted values must be kept.


# 1. Generate a 2-model ensemble (GBM + RF)

# Train & Cross-validate a GBM
my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train,
                  distribution = "bernoulli",
                  ntrees = 500,
                  max_depth = 3,
                  min_rows = 2,
                  learn_rate = 0.2,
                  nfolds = nfolds,
                  fold_assignment = "Modulo",
                  keep_cross_validation_predictions = TRUE,
                  seed = 1)

# Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train,
                          ntrees = 500,
                          nfolds = nfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)

# Train a stacked ensemble using the GBM and RF above
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                model_id = "my_ensemble_binomial",
                                base_models = list(my_gbm, my_rf))

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)

# Compare to base learner performance on the test set
perf_gbm_test <- h2o.performance(my_gbm, newdata = test)
perf_rf_test <- h2o.performance(my_rf, newdata = test)
baselearner_best_auc_test <- max(h2o.auc(perf_gbm_test), h2o.auc(perf_rf_test))
ensemble_auc_test <- h2o.auc(perf)
print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))

# 2. Generate a random grid of models and stack them together

# GBM parameters
learn_rate_opt <- c(0.01, 0.03)
max_depth_opt <- c(3, 4, 5, 6, 9)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 3,
                        seed = 1)

gbm_grid <- h2o.grid(algorithm = "gbm",
                     grid_id = "gbm_grid_binomial",
                     x = x,
                     y = y,
                     training_frame = train,
                     ntrees = 500,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

# Train a stacked ensemble using the GBM grid
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                model_id = "ensemble_gbm_grid_binomial",
                                base_models = gbm_grid@model_ids)

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)

# Compare to base learner performance on the test set
.getauc <- function(mm) h2o.auc(h2o.performance(h2o.getModel(mm), newdata = test))
baselearner_aucs <- sapply(gbm_grid@model_ids, .getauc)
baselearner_best_auc_test <- max(baselearner_aucs)
ensemble_auc_test <- h2o.auc(perf)
print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))


# another stacked ensemble can be semi-automated in H2O
# The H2O AutoML interface is designed to have as few parameters as possible so that all the user needs to do is point to their dataset, identify the response column and optionally specify a time constraint or limit on the number of total models trained.
# The current version of AutoML trains and cross-validates a default Random Forest (DRF), an Extremely Randomized Forest (XRT), a random grid of Gradient Boosting Machines (GBMs), a random grid of Deep Neural Nets, a fixed grid of GLMs. AutoML then trains two Stacked Ensemble models. Particular algorithms (or groups of algorithms) can be switched off using the exclude_algos argument. This is useful if you already have some idea of the algorithms that will do well on your dataset. As a recommendation, if you have really wide or sparse data, you may consider skipping the tree-based algorithms (GBM, DRF).
aml <- h2o.automl(x = x, y = y,
                  training_frame = train,
                  max_runtime_secs = 300)

# The AutoML object includes a “leaderboard” of models that were trained in the process, including the 5-fold cross-validated model performance (by default). 
# The number of folds used in the model evaluation process can be adjusted using the nfolds parameter. 
# If the user would like to score the models on a specific dataset, they can specify the leaderboard_frame argument, and then the leaderboard will show scores on that dataset instead.
lb <- aml@leaderboard
lb

# The models are ranked by a default metric based on the problem type (the second column of the leaderboard). In binary classification problems, that metric is AUC, and in multiclass classification problems, the metric is mean per-class error. In regression problems, the default sort metric is deviance. Some additional metrics are also provided, for convenience.
# The leader model is stored here
aml@leader




# deep learning

# The DL model will infer the response distribution from the response encoding if it is not specified explicitly through the distribution argument.
# H2O’s DL will not be reproducible if it is run on more than a single core, so in this example, the performance metrics below may vary slightly from what you see on your machine. The implementation uses a technique called “Hogwild!” which increases the speed of training at the cost of reproducibility on multiple cores.
# Early stopping (stop training before the specified number of epochs is completed to prevent overfitting) is enabled by default. If a validation frame is given, or if cross-validation is used (nfolds > 1), it will use validation error to determine the early stopping point. If just a training frame is given (and no CV), it will use the training set to perform early stopping. More on that below.

# we can use the h2o.grid() function to perform either a Cartesian or Randon Grid Search (RGS). Random Grid Search is usually a quicker way to find a good model, so we will provide a example of how to use H2O’s Random Grid Search on a DNN.
# see: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html
activation_opt <- c("Rectifier", "Maxout", "Tanh") # activation function choices
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01) # lasso
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01) # ridge

hyper_params <- list(activation = activation_opt, l1 = l1_opt, l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 300)

# The DL model will infer the response distribution from the response encoding if it is not specified explicitly through the distribution argument.
# H2O’s DL will not be reproducible if it is run on more than a single core, so in this example, the performance metrics below may vary slightly from what you see on your machine. The implementation uses a technique called “Hogwild!” which increases the speed of training at the cost of reproducibility on multiple cores.
# Early stopping (stop training before the specified number of epochs is completed to prevent overfitting) is enabled by default. If a validation frame is given, or if cross-validation is used (nfolds > 1), it will use validation error to determine the early stopping point. If just a training frame is given (and no CV), it will use the training set to perform early stopping. More on that below.
# Train the random grid. Fixed non-default parameters such as hidden=c(20,20) can be passed directly to the h2o.grid() function.
dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = train,
                    validation_frame = test,
                    seed = 1,
                    hidden = c(20,20),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)
dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
                           sort_by = "AUC", 
                           decreasing = TRUE)
print(dl_gridperf)

#Grab the model_id for the top DL model, chosen by AUC.

best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)

best_dl_perf <- h2o.performance(model = best_dl, newdata = test)
h2o.auc(best_dl_perf)

# remember to shutdown the cluster once we're done
h2o.shutdown(prompt = FALSE)


# for additional guidance about coding the ML algorithms used here, see: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html



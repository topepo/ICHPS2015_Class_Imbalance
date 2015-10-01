###################################################################
## Code for Workshop 4: Predictive Modeling on Data with Severe 
## Class Imbalance: Applications on Electronic Health Records. 	
## The course was conducted for the  International Conference on 
## Health Policy Statistics (ICHPS) on Wed, Oct 7, from 
## 10:15 AM - 12:15 PM.

###################################################################
## Example Data

load("emr.RData")

###################################################################
## Training/Test Split

library(caret)

set.seed(1732)
in_train <- createDataPartition(emr$Class, p = .75, list = FALSE)
training <- emr[ in_train,]
testing  <- emr[-in_train,]

mean(training$Class == "event")
mean(testing$Class == "event")

table(training$Class)
table(testing$Class)

###################################################################
## Overfitting to the Majority Class

library(partykit)
library(rpart)

rpart_small <- rpart(Class ~ ., data = training,
                    control = rpart.control(cp = 0.0062))

plot(as.party(rpart_small))

###################################################################
## Subsampling for class imbalances

## Define the resampling method and how we calculate performance

ctrl <- trainControl(method = "repeatedcv",
                     repeats = 5,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = twoClassSummary)

## Tune random forest models over this grid
mtry_grid <- data.frame(mtry = c(1:15, (4:9)*5))

###################################################################
## The basic random forest model with no adaptations

set.seed(1537)
rf_mod <- train(Class ~ ., 
                data = training,
                method = "rf",
                metric = "ROC",
                tuneGrid = mtry_grid,
                ntree = 1000,
                trControl = ctrl)

###################################################################
## This function is used to take the out of sample predictions and
## create an approximate ROC curve from them

roc_train <- function(object, best_only = TRUE, ...) {
  caret:::requireNamespaceQuietStop("pROC")
  caret:::requireNamespaceQuietStop("plyr")
  
  if(object$modelType != "Classification")
    stop("ROC curves are only availible for classification models")
  if(!any(names(object$modelInfo) == "levels"))
    stop(paste("The model's code is required to have a 'levels' module.",
               "See http://topepo.github.io/caret/custom_models.html#Components"))
  lvs <- object$modelInfo$levels(object$finalModel)
  if(length(lvs) != 2) 
    stop("ROC curves are only implemented here for two class problems")
  
  ## check for predictions
  if(is.null(object$pred)) 
    stop(paste("The out of sample predictions are required.",
               "See the `savePredictions` argument of `trainControl`"))
  
  if(best_only) {
    object$pred <- merge(object$pred, object$bestTune)
  }
  ## find tuning parameter names
  p_names <- as.character(object$modelInfo$parameters$parameter)
  p_combos <- object$pred[, p_names, drop = FALSE]
  
  ## average probabilities across resamples
  object$pred <- plyr::ddply(.data = object$pred, 
                             .variables = c("obs", "rowIndex", p_names),
                             .fun = function(dat, lvls = lvs) {
                               out <- mean(dat[, lvls[1]])
                               names(out) <- lvls[1]
                               out
                             })
  
  make_roc <- function(x, lvls = lvs, nms = NULL, ...) {
    out <- pROC::roc(response = x$obs,
                     predictor = x[, lvls[1]],
                     levels = rev(lvls))
    
    out$model_param <- x[1,nms,drop = FALSE]
    out
  }
  out <- plyr::dlply(.data = object$pred, 
                     .variables = p_names,
                     .fun = make_roc,
                     lvls = lvs,
                     nms = p_names)
  if(length(out) == 1)  out <- out[[1]]
  out
}

###################################################################
## Some plots of the data

ggplot(rf_mod)

plot(roc_train(rf_mod), 
     legacy.axes = TRUE,
     print.thres = .5,
     print.thres.pattern="   <- default %.1f threshold")

plot(roc_train(rf_mod), 
     legacy.axes = TRUE,
     print.thres.pattern = "Cutoff: %.2f (Sp = %.2f, Sn = %.2f)",
     print.thres = "best")

###################################################################
## Internal down-sampling

set.seed(1537)
rf_down_int <- train(Class ~ ., 
                     data = training,
                     method = "rf",
                     metric = "ROC",
                     strata = training$Class,
                     sampsize = rep(sum(training$Class == "event"), 2),
                     ntree = 1000,
                     tuneGrid = mtry_grid,
                     trControl = ctrl)

ggplot(rf_mod$results, aes(x = mtry, y = ROC)) + geom_point() + geom_line() + 
  geom_point(data = rf_down_int$results, aes(x = mtry, y = ROC), col = mod_cols[2]) + 
  geom_line(data = rf_down_int$results, aes(x = mtry, y = ROC), col = mod_cols[2]) + 
  theme_bw() + 
  xlab("#Randomly Selected Predictors") + 
  ylab("ROC (Repeated Cross-Validation)")

###################################################################
## External down-sampling

ctrl$sampling <- "down"
set.seed(1537)
rf_down <- train(Class ~ ., 
                 data = training,
                 method = "rf",
                 metric = "ROC",
                 tuneGrid = mtry_grid,
                 ntree = 1000,
                 trControl = ctrl)

geom_point(data = rf_down$results, aes(x = mtry, y = ROC), col = mod_cols[1]) + 
  geom_line(data = rf_down$results, aes(x = mtry, y = ROC), col = mod_cols[1]) + 
  theme_bw() + 
  xlab("#Randomly Selected Predictors") + 
  ylab("ROC (Repeated Cross-Validation)")

###################################################################
## Up-sampling

ctrl$sampling <- "up"
set.seed(1537)
rf_up <- train(Class ~ ., 
               data = training,
               method = "rf",
               tuneGrid = mtry_grid,
               ntree = 1000,
               metric = "ROC",
               trControl = ctrl)

ggplot(rf_mod$results, aes(x = mtry, y = ROC)) + geom_point() + geom_line() + 
  geom_point(data = rf_up$results, aes(x = mtry, y = ROC), col = mod_cols[3]) + 
  geom_line(data = rf_up$results, aes(x = mtry, y = ROC), col = mod_cols[3]) + 
  theme_bw() + 
  xlab("#Randomly Selected Predictors") + 
  ylab("ROC (Repeated Cross-Validation)")

###################################################################
## Up-sampling done **wrong**

ctrl2 <- trainControl(method = "repeatedcv",
                      repeats = 5,
                      classProbs = TRUE,
                      savePredictions = TRUE,
                      summaryFunction = twoClassSummary)
upped <- upSample(x = training[, -1], y = training$Class)

set.seed(1537)
rf_wrong <- train(Class ~ ., 
                  data = upped,
                  method = "rf",
                  tuneGrid = mtry_grid,
                  ntree = 1000,
                  metric = "ROC",
                  trControl = ctrl2)

ggplot(rf_mod$results, aes(x = mtry, y = ROC)) + geom_point() + geom_line() + 
  geom_point(data = rf_wrong$results, aes(x = mtry, y = ROC), col = mod_cols[3]) + 
  geom_line(data = rf_wrong$results, aes(x = mtry, y = ROC), col = mod_cols[3]) + 
  theme_bw() + 
  xlab("#Randomly Selected Predictors") + 
  ylab("ROC (Repeated Cross-Validation)")

###################################################################
## SMOTE 

ctrl$sampling <- "smote"
set.seed(1537)
rf_smote <- train(Class ~ ., 
                  data = training,
                  method = "rf",
                  tuneGrid = mtry_grid,
                  ntree = 1000,
                  metric = "ROC",
                  trControl = ctrl)

ggplot(rf_mod$results, aes(x = mtry, y = ROC)) + geom_point() + geom_line() + 
  geom_point(data = rf_smote$results, aes(x = mtry, y = ROC), col = mod_cols[4]) + 
  geom_line(data = rf_smote$results, aes(x = mtry, y = ROC), col = mod_cols[4]) + 
  theme_bw() + 
  xlab("#Randomly Selected Predictors") + 
  ylab("ROC (Repeated Cross-Validation)")

###################################################################
## Make code to measure performance for cost-sensitive learning


fourStats <- function (data, lev = levels(data$obs), model = NULL) {
  accKapp <- postResample(data[, "pred"], data[, "obs"])
  out <- c(accKapp,
           sensitivity(data[, "pred"], data[, "obs"], lev[1]),
           specificity(data[, "pred"], data[, "obs"], lev[2]))
  names(out)[3:4] <- c("Sens", "Spec")
  out
}

ctrl_cost <- trainControl(method = "repeatedcv",
                          repeats = 5,
                          classProbs = FALSE,
                          savePredictions = TRUE,
                          summaryFunction = fourStats)

###################################################################
## Setup a custom tuning grid by first fitting a rpart model and
## getting the unique Cp values

rpart_init <- rpart(Class ~ ., data = training, cp = 0)$cptable

cost_grid <- expand.grid(cp = rpart_init[, "CP"],
                         Cost = 1:5)
set.seed(1537)
rpart_costs <- train(Class ~ ., data = training, 
                     method = "rpartCost",
                     tuneGrid = cost_grid,
                     metric = "Kappa",
                     trControl = ctrl_cost)

ggplot(rpart_costs) + 
  scale_x_log10(breaks = 10^pretty(log10(rpart_costs$results$cp), n = 5)) + 
  theme(legend.position = "top")

###################################################################
## C5.0 with costs

cost_grid <- expand.grid(trials = 1:3,
                         winnow = FALSE,
                         model = "tree",
                         cost = seq(1, 10, by = .25))
set.seed(1537)
c5_costs <- train(Class ~ ., data = training, 
                  method = "C5.0Cost",
                  tuneGrid = cost_grid,
                  metric = "Kappa",
                  trControl = ctrl_cost)

c5_costs_res <- subset(c5_costs$results, trials <= 3)
c5_costs_res$trials <- factor(c5_costs_res$trials)


ggplot(c5_costs_res, aes(x = cost, y = Kappa, group = trials)) +
  geom_point(aes(color = trials)) + 
  geom_line(aes(color = trials)) + 
  ylab("Kappa (Repeated Cross-Validation)")+ 
  theme(legend.position = "top")


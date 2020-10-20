setwd("~/Boruta-RFE-R-main/Final model")

# Load data 
v3data.train <- read.csv("full_train_setcsv.csv")
v3data.test <-  read.csv("full_test_setcsv.csv")

# Load caret package
library(caret)
names(getModelInfo()) # Check model name in caret 

# Model training 
set.seed(1234)
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 2, allowParallel = TRUE)


## Parallel Training
library(doParallel)
cluster <- makeCluster(detectCores(logical = TRUE)) # leave one CPU spare...
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(caret)
})


# Linear Regression 
set.seed(1234)
model_LR<-train(new_cases ~ . ,data=v3data.train, method='lm', trControl=fitControl,tuneLength=10)
print(model_LR)
plot(model_LR)
saveRDS(model_rf, "linear_reg_model.rds")



# CART Decision Tree
set.seed(1234)
model_CART<-train(new_cases ~ . ,data=v3data.train, method='rpart2', trControl=fitControl,tuneLength=10)
print(model_CART)
plot(model_CART)
rfImp_CART <- varImp(model_CART) 
plot(rfImp_CART, top = 53)
saveRDS(model_rf, "CART_model.rds")



# M5 Model tree
set.seed(1234)
model_M5<-train(new_cases ~ . ,data=v3data.train, method='M5', trControl=fitControl,tuneLength=10)
print(model_M5)
plot(model_M5)
rfImp_M5 <- varImp(model_M5) 
plot(rfImp_M5, top = 53)
saveRDS(model_rf, "M5_model.rds")



# Neural Network
set.seed(1234)
model_NN<-train(new_cases ~ .,data=v3data.train, method='neuralnet', trControl=fitControl,tuneLength=10)
print(model_NN)
plot(model_NN)
rfImp_NN <- varImp(model_NN) 
plot(rfImp_NN, top = 53)

# Random Forest
set.seed(1234)
model_rf<-train(new_cases ~ . ,data=v3data.train, method='rf', trControl=fitControl,tuneLength=10, importance = T)
print(model_rf)
plot(model_rf)
rfImp_rf <- varImp(model_rf) 
plot(rfImp_rf, top = 53)


# Gradient Boosted Tree model
set.seed(1234)
model_gbm<-train(new_cases ~ . ,data=v3data.train, method='gbm', trControl=fitControl,tuneLength=10, importance = T)
print(model_gbm)
plot(model_gbm)
rfImp_gbm <- varImp(model_gbm) 
plot(rfImp_gbm, top = 53)
saveRDS(model_rf, "GBT_model.rds")


### Stacking model
# collect resamples
results <- resamples(list(CART=model_CART, 
                          RF=model_rf, 
                          LR=model_LR))


# summarize differences between modes
summary(results)

# box and whisker plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)


# dot plots of accuracy
scales <- list(x=list(relation="free"), y=list(relation="free"))
p <- dotplot(results, scales=scales)
stripchart(results, vertical=TRUE, add=TRUE, method="stack", col='red', pch="*")
p + stat_summary(fun.data="mean_sdl", fun.args = list(mult=1), 
                 geom="crossbar", width=0.5)
summary(p)


### Plotting Varianle Importance each model 
plot(varImp(object=model_gbm),top = 11, main="GBM - Variable Importance")
plot(varImp(object=model_rf),top = 50,main="RF - Variable Importance")
plot(varImp(object=model_NN),top = 11, main="neuralnet - Variable Importance")
plot(varImp(object=model_M5),top = 11, main="M5 - Variable Importance")
plot(varImp(object=model_CART),top = 11,main="CART (Boruta) - Variable Importance")

names(v3data.test)
### Predictions in test data (If model_RF has highest accuracy)

v3data.train$Predicted_new_cases <-predict.train(object=model_rf,v3data.train)

v3data.test$Predicted_new_cases <-predict.train(object=model_rf,v3data.test)


### Save Trained Model
saveRDS(model_rf, "model.rds")
model_rf <- readRDS("full_var_model.rds")

### Error measurement

library("Metrics")

# Training error

mse(v3data.train$Predicted_new_cases , v3data.train$new_cases)
rmse(v3data.train$Predicted_new_cases , v3data.train$new_cases)
postResample(pred= v3data.train$Predicted_new_cases ,obs =v3data.train$new_cases)
cor(v3data.train$Predicted_new_cases , v3data.train$new_cases)	

# Testing error
mse(v3data.test$Predicted_new_cases , v3data.test$new_cases)
rmse(v3data.test$Predicted_new_cases , v3data.test$new_cases)
postResample(pred= v3data.test$Predicted_new_cases ,obs =v3data.test$new_cases)
cor(v3data.test$Predicted_new_cases , v3data.test$new_cases)	


# Export data

write.csv(v3data.train, ".....csv", row.names = FALSE)

write.csv(v3data.test, "......csv", row.names = FALSE)




# H-staistic
library(dplyr)     # basic data transformation
library(iml)       # ML interprtation
library(h2o)       # machine learning modeling
library(stats)
library(randomForest)
# 1. create a data frame with just the features
features <- as.data.frame(v3data.train) %>% select(-new_cases)

# 2. Create a vector with the actual responses
response <- as.numeric(as.vector(v3data.train$new_cases))


# initialize h2o session
h2o.no_progress()
h2o.init()


## Parallel Training
library(doParallel)
cluster <- makeCluster(detectCores(logical = TRUE)) # leave one CPU spare...
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(iml)
})
# 3. Create custom predict function that returns the predicted values as a
#    vector (probability of purchasing in our example)
pred <- function(model, newdata)  {
  results <- as.data.frame(h2o.predict(model, as.h2o(newdata)))
  return(results[[3L]])
}

# example of prediction output
pred(model_rf, features) %>% head()

predictor.rf <- Predictor$new(
  model = model_rf, 
  data = features, 
  y = response, 
  predict.fun = pred,
  class = "classification"
)

interact.rf  <- Interaction$new(predictor.rf) %>% plot() + ggtitle("RF")

interact.rf

## Parallel
library("future")
library("future.callr")
plan("callr", workers = 6)


## Parallel Training
library(doParallel)
cluster <- makeCluster(detectCores(logical = TRUE)) # leave one CPU spare...
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(future.callr)
})

## Two way Interaction
names(v3data.test)
two_way_interact.rf  <- Interaction$new(predictor.rf, feature = "c7_2_action") %>% plot() 


----------------------------------------------------------
## Parallel Training
library(doParallel)
cluster <- makeCluster(detectCores(logical = TRUE)) # leave one CPU spare...
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(pdp)
})


library(pdp)
## Compute pdp for single variable 
pd_1st_order <- partial(model_rf, pred.var = c("c7_2_action"), rug = TRUE, plot = TRUE)

pd_1st_order

## Compute contour pdp for 2 variables 
pd <- partial(model_rf, pred.var = c("c7_1_action", "c7_2_action"), chull = FALSE, contour = TRUE)
# Add contour lines and use a different color palette
rwb <- colorRampPalette(c("blue", "white", "red" ))
pdp_2nd_order <- plotPartial(pd, contour = TRUE, col.regions = rwb)

pdp_2nd_order








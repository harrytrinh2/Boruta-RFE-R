
# Load data 
setwd('c://Users/dungb/Documents')
v3data.train <- read.csv("80-20/full_train_setcsv.csv")
v3data.test  <- read.csv("80-20/full_test_setcsv.csv")

names(v3data.test)

head(v3data.train)
### Load Trained Model
my_model <- readRDS("80-20/full_var_model.rds")



# Load caret package
library(caret)
names(getModelInfo()) # Check model name in caret 


### Plotting Varianle Importance each model 
plot(varImp(object=model_gbm),top = 11, main="GBM - Variable Importance")
plot(varImp(object=my_model),top = 11,main="RF - Variable Importance")
plot(varImp(object=model_NN),top = 11, main="neuralnet - Variable Importance")
plot(varImp(object=model_M5),top = 11, main="M5 - Variable Importance")
plot(varImp(object=model_CART),top = 11,main="CART (Boruta) - Variable Importance")


### Predictions in test data (If model_RF has highest accuracy)

v3data.train$Predicted_Output <-predict.train(object=my_model,v3data.train)

v3data.test$Predicted_Output <-predict.train(object=my_model,v3data.test)

### Error measurement

library("Metrics")

# Training error

mse(v3data.train$Predicted_Output , v3data.train$new_cases)
rmse(v3data.train$Predicted_Output , v3data.train$new_cases)
postResample(pred= v3data.train$Predicted_Output ,obs =v3data.train$new_cases)
cor(v3data.train$Predicted_Output , v3data.train$new_cases)	

# Testing error
mse(v3data.test$Predicted_Output , v3data.test$new_cases)
rmse(v3data.test$Predicted_Output , v3data.test$new_cases)
postResample(pred= v3data.test$Predicted_Output ,obs =v3data.test$new_cases)
cor(v3data.test$Predicted_Output , v3data.test$new_cases)	



# H-staistic
library(dplyr)     # basic data transformation
library(iml)       # ML interprtation
library(h2o)       # machine learning modeling
library(stats)

# 1. create a data frame with just the features
features <- as.data.frame(v3data.train) %>% select(-new_cases)

# 2. Create a vector with the actual responses
response <- as.numeric(as.vector(v3data.train$new_cases))


# initialize h2o session
h2o.no_progress()
h2o.init()

# 3. Create custom predict function that returns the predicted values as a
#    vector (probability of purchasing in our example)
pred <- function(model, newdata)  {
  results <- as.data.frame(h2o.predict(model, as.h2o(newdata)))
  return(results[[3L]])
}

# example of prediction output
pred(my_model, features) %>% head()

predictor.rf <- Predictor$new(
  model = my_model, 
  data = features, 
  y = response, 
  predict.fun = pred,
  class = "classification"
)

interact.rf  <- Interaction$new(predictor.rf) %>% plot() + ggtitle("RF")




----------------------------------------------------------
  
library(pdp)
## Compute pdp for single variable 
pd_1st_order <- partial(my_model, pred.var = c("C7_2_action"), rug = TRUE, plot = TRUE)


pd_1st_order

## Compute contour pdp for 2 variables 
pd <- partial(my_model, pred.var = c("C7_1_action", "C7_2_action"), chull = FALSE, contour = TRUE)
# Add contour lines and use a different color palette
rwb <- colorRampPalette(c("blue", "white", "red" ))
pdp_2nd_order <- plotPartial(pd, contour = TRUE, col.regions = rwb)

pdp_2nd_order







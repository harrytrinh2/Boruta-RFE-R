
# Load data 
v3data.train 
v3data.test

# Load caret package
library(caret)
names(getModelInfo()) # Check model name in caret 

# Model training 
set.seed(1234)
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 2)

# Linear Regression 
set.seed(1234)
model_LR<-train(Output ~ . ,data=v3data.train, method='lm', trControl=fitControl,tuneLength=10)
print(model_LR)
plot(model_LR)

# CART Decision Tree
set.seed(1234)
model_CART<-train(Output ~ . ,data=v3data.train, method='rpart2', trControl=fitControl,tuneLength=10)
print(model_CART)
plot(model_CART)
rfImp_CART <- varImp(model_CART) 
plot(rfImp_CART, top = 53)

# M5 Model tree
set.seed(1234)
model_M5<-train(Output ~ . ,data=v3data.train, method='M5', trControl=fitControl,tuneLength=10)
print(model_M5)
plot(model_M5)
rfImp_M5 <- varImp(model_M5) 
plot(rfImp_M5, top = 53)

# Neural Network
set.seed(1234)
model_NN<-train(Output ~ .,data=v3data.train, method='neuralnet', trControl=fitControl,tuneLength=10)
print(model_NN)
plot(model_NN)
rfImp_NN <- varImp(model_NN) 
plot(rfImp_NN, top = 53)

# Random Forest
set.seed(1234)
model_rf<-train(Output ~ . ,data=v3data.train, method='rf', trControl=fitControl,tuneLength=10, importance = T)
print(model_rf)
plot(model_rf)
rfImp_rf <- varImp(model_rf) 
plot(rfImp_rf, top = 53)


# Gradient Boosted Tree model
set.seed(1234)
model_gbm<-train(Output ~ . ,data=v3data.train, method='gbm', trControl=fitControl,tuneLength=10, importance = T)
print(model_gbm)
plot(model_gbm)
rfImp_gbm <- varImp(model_gbm) 
plot(rfImp_gbm, top = 53)

### Stacking model
# collect resamples
results <- resamples(list(CART=model_CART, RF=model_rf, GBM=model_gbm, M5=model_M5, LR=model_LR, NeuraNet = model_NN))


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
plot(varImp(object=model_rf),top = 11,main="RF - Variable Importance")
plot(varImp(object=model_NN),top = 11, main="neuralnet - Variable Importance")
plot(varImp(object=model_M5),top = 11, main="M5 - Variable Importance")
plot(varImp(object=model_CART),top = 11,main="CART (Boruta) - Variable Importance")


### Predictions in test data (If model_RF has highest accuracy)

v3data.train$Predicted_Output <-predict.train(object=model_RF,v3data.train)

v3data.test$Predicted_Output <-predict.train(object=model_RF,v3data.test)

### Error measurement

library("Metrics")

# Training error

mse(v3data.train$Predicted_Output , v3data.train$Output)
rmse(v3data.train$Predicted_Output , v3data.train$Output)
postResample(pred= v3data.train$Predicted_Output ,obs =v3data.train$Output)
cor(v3data.train$Predicted_Output , v3data.train$Output)	

# Testing error
mse(v3data.test$Predicted_Output , v3data.test$Output)
rmse(v3data.test$Predicted_Output , v3data.test$Output)
postResample(pred= v3data.test$Predicted_Output ,obs =v3data.test$Output)
cor(v3data.test$Predicted_Output , v3data.test$Output)	


# Export data

write.csv(v3data.train, ".....csv", row.names = FALSE)

write.csv(v3data.test, "......csv", row.names = FALSE)


----------------------------------------------------------
  
  library(pdp)
## Compute pdp for single variable 
pd_1st_order <- partial(model_rf, pred.var = c("C1_3_days"), rug = TRUE, plot = TRUE)

pd_1st_order

## Compute contour pdp for 2 variables 
pd <- partial(model_rf, pred.var = c("C3_0_days", "C3_2_days"), chull = FALSE, contour = TRUE)
# Add contour lines and use a different color palette
rwb <- colorRampPalette(c("blue", "white", "red" ))
pdp_2nd_order <- plotPartial(pd, contour = TRUE, col.regions = rwb)

pdp_2nd_order







library(iml)
library(randomForest)

setwd('c://Users/dungb/Documents')
data <- read.csv("80-20/train_set.csv")
X_train <- read.csv("80-20/trainR.csv")
y_train <- read.csv("80-20/ytrainR.csv")

head(data)

library(dplyr)
v3data.train <- dplyr::select(v3data.train, c())

head(v3data.train)
names(v3data.train)
typeof(v3data.train)


set.seed(131)
rf <- randomForest(new_cases ~ ., data=data, mtry=3,
                   importance=TRUE, na.action=na.omit)



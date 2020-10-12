
setwd('Documents/GlobalCovid19/Model_tuning_prediction/')
## Load train data 
v3data.train <- read.csv("official_train_test_split/80-20/train_set.csv")

#### Step 1.  Remember to check multicolineatity to remove highly correlated variables 

# Define stepwise VIF function 
vif_func<-function(in_frame,thresh=10,trace=T,...){
  library(fmsb)
  if(any(!'data.frame' %in% class(in_frame))) in_frame<-data.frame(in_frame)
  #get initial vif value for all comparisons of variables
  vif_init<-NULL
  var_names <- names(in_frame)
  for(val in var_names){
    regressors <- var_names[-which(var_names == val)]
    form <- paste(regressors, collapse = '+')
    form_in <- formula(paste(val, '~', form))
    vif_init<-rbind(vif_init, c(val, VIF(lm(form_in, data = in_frame, ...))))
  }
  vif_max<-max(as.numeric(vif_init[,2]), na.rm = TRUE)
  
  if(vif_max < thresh){
    if(trace==T){ #print output of each iteration
      prmatrix(vif_init,collab=c('var','vif'),rowlab=rep('',nrow(vif_init)),quote=F)
      cat('\n')
      cat(paste('All variables have VIF < ', thresh,', max VIF ',round(vif_max,2), sep=''),'\n\n')
    }
    return(var_names)
  }
  else{
    
    in_dat<-in_frame
    
    #backwards selection of explanatory variables, stops when all VIF values are below 'thresh'
    while(vif_max >= thresh){
      
      vif_vals<-NULL
      var_names <- names(in_dat)
      
      for(val in var_names){
        regressors <- var_names[-which(var_names == val)]
        form <- paste(regressors, collapse = '+')
        form_in <- formula(paste(val, '~', form))
        vif_add<-VIF(lm(form_in, data = in_dat, ...))
        vif_vals<-rbind(vif_vals,c(val,vif_add))
      }
      max_row<-which(vif_vals[,2] == max(as.numeric(vif_vals[,2]), na.rm = TRUE))[1]
      
      vif_max<-as.numeric(vif_vals[max_row,2])
      
      if(vif_max<thresh) break
      
      if(trace==T){ #print output of each iteration
        prmatrix(vif_vals,collab=c('var','vif'),rowlab=rep('',nrow(vif_vals)),quote=F)
        cat('\n')
        cat('removed: ',vif_vals[max_row,1],vif_max,'\n\n')
        flush.console()
      }
      
      in_dat<-in_dat[,!names(in_dat) %in% vif_vals[max_row,1]]
      
    }
    
    return(names(in_dat))
    
  }
  
}

# Check VIF

df.subset <- subset(v3data.train, select= -c(new_cases, weather_situation, isHoliday, is_weekend, Day_of_Week, continent))

vif_result = vif_func(in_frame=df.subset ,thresh=10,trace=T)

# Remove variable with VIF > 10 from the dataset v3data.train
names(v3data.train)
drops <- c('CountryCode',"population_density",'cvd_death_rate','gdp_per_capita','diabetes_prevalence','life_expectancy','aged_65_older_sum','urbanPopulation','healthExpenditure','h1_2_action','Days.since.first.case','c8_3_action','h3_2_action')
v3data.train =v3data.train[ , !(names(v3data.train) %in% drops)]
names(v3data.train)


#### Feature selection by RFE

library(caret)
library(randomForest)
library(doParallel)
Mycluster = makeCluster(detectCores())
registerDoParallel(Mycluster)
set.seed(143)
control <- rfeControl(functions=rfFuncs, method="repeatedcv", number=10, repeats=1, allowParallel = TRUE)

library(dplyr)
v3data.train <- dplyr::select(v3data.train, c("record_date",                          
                                              "c1_0_action","c1_1_action",                          
                                              "c1_2_action","c1_3_action",                          
                                              "c2_0_action","c2_1_action",                          
                                              "c2_2_action","c2_3_action",                          
                                              "c3_0_action","c3_1_action",                          
                                              "c3_2_action","c4_0_action",                          
                                              "c4_1_action","c4_2_action",                          
                                              "c4_3_action","c4_4_action",                          
                                              "c5_0_action","c5_1_action",                          
                                              "c5_2_action","c6_0_action",                          
                                              "c6_1_action","c6_2_action",                          
                                              "c6_3_action","c7_0_action",                          
                                              "c7_1_action","c7_2_action",                          
                                              "c8_0_action","c8_1_action",                          
                                              "c8_2_action","c8_4_action",                          
                                              "e1_0_action","e1_1_action",                          
                                              "e1_2_action","e2_0_action",                          
                                              "e2_1_action","e2_2_action",                          
                                              "h1_0_action","h1_1_action",                          
                                              "h2_0_action","h2_1_action",                          
                                              "h2_2_action","h2_3_action",                          
                                              "h3_0_action","h3_1_action",                          
                                              "E3_Fiscal.measures","E4_International.support",             
                                              "H4_Emergency.investment.in.healthcare", "H5_Investment.in.vaccines",            
                                              "humidity","weather_situation","temperature",                          
                                              "windSpeed",  "Number.of.Tweet",                      
                                              "Sentiments", "isHoliday", 
                                              "Day_of_Week","is_weekend",
                                              "continent","new_cases"))

head(v3data.train)
names(v3data.train)
typeof(v3data.train)

## task 1 failed - "Can not handle categorical predictors with more than 53 categories."
## delete more 
drops <- c('record_date')
v3data.train =v3data.train[ , !(names(v3data.train) %in% drops)]
names(v3data.train)
rfe.train <- rfe(v3data.train[,1:58],v3data.train[,59], sizes=1:58, rfeControl=control) 


rfe.train
plot(rfe.train, type=c("g", "o"), cex = 1.0, col = 1:11)
predictors(rfe.train)
names(rfe.train)
rfe.train.optVariables


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

names(df.subset)

vif_result = vif_func(in_frame=df.subset ,thresh=10,trace=T)

# Remove variable with VIF > 10 from the dataset v3data.train

names(v3data.train)

drops <- c('CountryCode',"population_density",'cvd_death_rate','gdp_per_capita','diabetes_prevalence','life_expectancy','aged_65_older_sum','urbanPopulation','healthExpenditure','h1_2_action','Days.since.first.case','c8_3_action','h3_2_action')
v3data.train =v3data.train[ , !(names(v3data.train) %in% drops)]

names(v3data.train)

#### Feature selection by Boruta

library(Boruta)
set.seed(123)
names(v3data.train)

#v3data.train[-c("diabetes_prevalence")]
#drops <- c("diabetes_prevalence")
#v3data.train =v3data.train[ , !(names(v3data.train) %in% drops)]

boruta.train <- Boruta(new_cases ~. , data = v3data.train, doTrace = 2)

print(boruta.train)

plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)


final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

getSelectedAttributes(final.boruta, withTentative = F)

boruta.df <- attStats(final.boruta)
class(boruta.df)
print(boruta.df)
# This code is used for two stage stacking 
# Xun Qiu
# 2016-08-09

# Input: from step 3 in code-xgb.R
# - y  (target value)
# - x_all (training+test)
# - device_gender 
# - group_name
# - id (device_id for training+test)

library(caret)
library(xgboost)

## -------------- 1. Initialization ------------------------------------
## ---------------------------------------------------------------------
# preparing inputs for training xgb model
idx_train <- which(!is.na(y))                                                  
idx_test <- which(is.na(y))
train_data <- x_all[idx_train,]
test_data <- x_all[idx_test,]
train_label <- match(y[idx_train],group_name)-1
test_label<- match(y[idx_test],group_name)-1
dtest_stage1 <- xgb.DMatrix(test_data,label=test_label,missing=NA)
dtrain_stage1 <- xgb.DMatrix(train_data,label=train_label,missing=NA)

# Create folds for generating stage-1 prediction
set.seed(314)
cvindex <- createFolds(y[idx_train],k=5)


## -------------- 2. stage 1 training ----------------------------------
## ---------------------------------------------------------------------
## model1, gblinear (single model as in code-xgb.R) 
param1 <- list(booster="gblinear",
               num_class=length(group_name),
               objective="multi:softprob",
               eval_metric="mlogloss",
               eta=0.01,
               lambda=5,
               lambda_bias=0,
               alpha=2)
ntree1 <- 212
seed1 <- 314

trainmodel(param1, ntree1, seed1, modelnr=1, 
           train_data, train_label,cvindex, dtest_stage1, 
           id, idx_train, idx_test,group_name)

## model2, gbtree (code-xgb.R)
param2 <- list(booster="gbtree",
               num_class=12,
               objective="multi:softprob",
               eval_metric="mlogloss",
               eta=0.05,
               max_depth = 5,
               gamma = 4,
               subsample = 0.7,
               colsample_bytree = 0.7,
               min_child_weight = 7)

ntree2 <- 1158
seed2 <- 8548

trainmodel(param2, ntree2, seed2, modelnr=2, 
           train_data, train_label,cvindex, dtest_stage1, 
           id, idx_train, idx_test,group_name)


## -------------- 3. Preparing input for stage 2------------------------
## ---------------------------------------------------------------------
# read in training data from stage1
train_stage1_1 <- read.csv(file="stacking/train_data_stage1_1.csv", 
                           header=T, 
                           numerals = "no.loss")
train_stage1_2 <- read.csv(file="stacking/train_data_stage1_2.csv", 
                           header=T, 
                           numerals = "no.loss")

colnames(train_stage1_1) <- c("device_id",group_name)
colnames(train_stage1_2) <- c("device_id",group_name)

# read in test data from stage1
test_stage1_1 <- read.csv(file="stacking/test_stage1_1.csv",
                          header=T,
                          numerals = "no.loss")
test_stage1_2 <- read.csv(file="stacking/test_stage1_2.csv",
                          header=T,
                          numerals = "no.loss")

colnames(test_stage1_1) <- c("device_id",group_name)
colnames(test_stage1_2) <- c("device_id",group_name)


# combine training data for stage2
train_data_stage2 <- cbind(train_stage1_1,
                           train_stage1_2[,-1])

# combine test data for stage2
test_data_stage2 <- cbind(test_stage1_1, 
                          test_stage1_2[,-1])


## -------------- 4. stage 2 training ----------------------------------
## ---------------------------------------------------------------------
train_label_stage2 <- train_label[c(cvindex$Fold1,
                                    cvindex$Fold2,
                                    cvindex$Fold3,
                                    cvindex$Fold4,
                                    cvindex$Fold5)]

# preparing input for xgb model
train_data_stage2_matrix <- as.matrix(train_data_stage2[,-1])
test_data_stage2_matrix <- as.matrix(test_data_stage2[,-1])

dtrain <- xgb.DMatrix(train_data_stage2_matrix,
                      label=train_label_stage2,
                      missing=NA)
dtest <- xgb.DMatrix(test_data_stage2_matrix,
                     label=test_label,
                     missing=NA)

param_stage2 <- list(booster="gblinear",
                     num_class=length(group_name),
                     objective="multi:softprob",
                     eval_metric="mlogloss",
                     eta=0.1,
                     lambda=2,
                     lambda_bias=0,
                     alpha=0.1)

watchlist <- list(train=dtrain)

set.seed(314)
fit_cv <- xgb.cv(params=param_stage2,
                 data=dtrain,
                 nrounds=1000,
                 watchlist=watchlist,
                 nfold=5,
                 early.stop.round=10,
                 verbose=1)

ntree <- 355
set.seed(314)
fit_xgb <- xgb.train(params=param_stage2,
                     data=dtrain,
                     nrounds=ntree,
                     watchlist=watchlist,
                     verbose=1)


## -------------- 5. final prediction ----------------------------------
## ---------------------------------------------------------------------
pred <- predict(fit_xgb,dtest)
pred_detail <- t(matrix(pred,nrow=length(group_name)))
res_submit <- cbind(id=id[idx_test],as.data.frame(pred_detail))
colnames(res_submit) <- c("device_id",group_name)
write.csv(res_submit,file="submission_stacking.csv",row.names=F,quote=F)


## -------------- Auxiliary functions ----------------------------------
## ---------------------------------------------------------------------
## Get predictions from stage1 model, both for training data and test data
trainmodel <- function(params, ntree=100, seed=314, modelnr=1, 
                       train_data, train_label,cvindex, dtest_stage1, 
                       id, idx_train, idx_test,group_name)
{
  train_stage1 <- data.frame(matrix(ncol=13,nrow=0))
  colnames(train_stage1) <- c("device_id",group_name)
  
  for (i in c(1:5))
  {
    command <- paste("train_data[-cvindex$Fold",i,",]",sep="")
    eval(parse(text=paste("train_data_touse","<-",command,sep="")))
    
    command <- paste("train_label[-cvindex$Fold",i,"]",sep="")
    eval(parse(text=paste("train_label_touse","<-",command,sep="")))
    
    command <- paste("train_data[cvindex$Fold",i,",]",sep="")
    eval(parse(text=paste("val_data_touse","<-",command,sep="")))
    
    command <- paste("train_label[cvindex$Fold",i,"]",sep="")
    eval(parse(text=paste("val_label_touse","<-",command,sep="")))
    
    command <- paste("idx_train[cvindex$Fold",i,"]",sep="")
    eval(parse(text=paste("idx_val","<-",command,sep="")))
    
    dtrain <- xgb.DMatrix(train_data_touse,label=train_label_touse,missing=NA)
    dval <- xgb.DMatrix(val_data_touse,label=val_label_touse,missing=NA)
    
    dtrain1 <- dtrain
    watchlist1 <- list(train=dtrain1)
    set.seed(seed)
    fit_xgb_1 <- xgb.train(params=params,
                           data=dtrain1,
                           nrounds=ntree,
                           watchlist=watchlist1,
                           verbose=1)
    
    pred <- predict(fit_xgb_1,dval)
    pred_detail <- t(matrix(pred,nrow=length(group_name)))
    train_stage1_tmp <- cbind(id=id[idx_val],as.data.frame(pred_detail))
    colnames(train_stage1_tmp) <- c("device_id",group_name)
    
    train_stage1 <- rbind(train_stage1, train_stage1_tmp)
  }
  
  filename <- paste("stacking/train_data_stage1_", modelnr, ".csv", sep="")
  write.csv(file=filename, train_stage1, row.names = F)
  
  # testing data for stage2
  dtrain1 <- dtrain_stage1
  set.seed(seed)
  fit_xgb_1 <- xgb.train(params=params,
                         data=dtrain1,
                         nrounds=ntree,
                         watchlist=watchlist1,
                         verbose=1)
  
  pred <- predict(fit_xgb_1,dtest_stage1)
  pred_detail <- t(matrix(pred,nrow=length(group_name)))
  test_stage1_1 <- cbind(id=id[idx_test],as.data.frame(pred_detail))
  colnames(test_stage1_1) <- c("device_id",group_name)
  
  filename <- paste("stacking/test_stage1_", modelnr, ".csv", sep="")
  write.csv(file=filename,test_stage1_1,row.names = F)
  
}


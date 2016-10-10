## This code is for building single model with XGBoost
## Kaggle competition : Talking data
## Features: 
## - phone_brand/device_model/bag of apps_installed/bag of labels
## - phonebrand_group1/foreign brand/old brand
## Xun Qiu
## 2016-08

# results:
# xgblinear: 2.2727, ntree=212
# xgbtree: 2.278， ntree=1158

## Load libraries
library(data.table)
library(Matrix)
library(xgboost)

## Set global options
options(stringsAsFactors=F,scipen=99)

## Auxiliary function
f_getunique <- function(z){paste(unique(unlist(strsplit(z,","))),
                                 collapse=",")}

## ------------------1. Pre-processsing----------------------------------------
## ----------------------------------------------------------------------------
## Apps
events <- fread("events.csv",
                colClasses=c("character","character","character",              
                             "numeric","numeric"))
events <- unique(events[, list(device_id,event_id)])
setkeyv(events,c("device_id","event_id"))

event_app <- fread("app_events.csv", colClasses=rep("character",4))
setkey(event_app,event_id)

# each row is one event_id with all its app_id
event_apps <- event_app[, list(apps=paste(unique(app_id),collapse=",")),       
                        by="event_id"]

device_event_apps <- merge(events,event_apps,by="event_id")

# each row is one device_id with all its app_id
device_apps <- device_event_apps[,list(apps=f_getunique(apps)),
                                 by="device_id"]

tmp <- strsplit(device_apps$apps,",")
# each row is one device_id with one app_id
device_apps <- data.table(device_id = rep(device_apps$device_id, 
                                          times=sapply(tmp,length)),
                          app_id = unlist(tmp))

write.csv(file="device_apps_installed.csv", device_apps, row.names = F)
rm(tmp, device_event_apps, event_apps, events, event_app)
gc()

## Labels
setkeyv(device_apps,c("device_id","app_id"))
device_apps <- unique(device_apps[,list(device_id, app_id)],by=NULL)

app_label <- fread("app_labels.csv",
                   colClasses=c("character","character"))
setkey(app_label,app_id)

# each row is one app_id with all its label_id
app_labels <- app_label[,list(labels=paste(unique(label_id),collapse = ",")),
                        by="app_id"]

device_app_labels <- merge(device_apps,app_labels,by="app_id")

# each row is one device_id with all its label_id
device_labels <- device_app_labels[,list(labels=f_getunique(labels)),
                                   by="device_id"]

tmp <- strsplit(device_labels$labels,",")
# each row is one device_id and one label_id
device_labels <- data.table(device_id=rep(device_labels$device_id,
                                          times=sapply(tmp,length)),
                            label_id=unlist(tmp))

write.csv(file="device_labels_installed.csv",device_labels,row.names = F)
rm(tmp, device_app_labels,device_apps, app_labels, app_label)
gc()


## ------------------2. Feature Engineering -----------------------------------
## ----------------------------------------------------------------------------
## Device
device <- read.csv(file="phone_brand_device_model.csv",
                   encoding="UTF-8",
                   header=T,
                   numerals="no.loss") 

device <- unique(device)
device <- device[-which(duplicated(device$device_id)),]

Sys.setlocale(category = "LC_ALL", locale = "chs")

device$device_model_new <- paste(device$phone_brand,
                                 device$device_model,
                                 sep="_")
device$device_model_new<-as.factor(device$device_model_new)

genderage_train <- read.csv(file="gender_age_train.csv",
                            header=T, 
                            numerals="no.loss")

genderage_test <- read.csv(file="gender_age_test.csv",
                           header=T, 
                           numerals="no.loss")
genderage_test$gender <- NA
genderage_test$age <- NA
genderage_test$group <- NA
genderage <- rbind(genderage_train, genderage_test)

devicegender <- merge(genderage, device, by="device_id")

devicegender_train <- devicegender[devicegender$device_id %in% 
                                     genderage_train$device_id,]
rm(device)
gc()

# Feature1: phone brand group1 based on popularity
# quantile(devicegender_train$phone_brand, probs=seq(0,1,0.05))
tmp <- as.data.frame(table(devicegender_train$phone_brand))
devicegender$phonebrand_group1<-as.character(devicegender$phone_brand)
brand_popular <- tmp$Var1[tmp$Freq>=3015]
devicegender$phonebrand_group1[devicegender$phone_brand %in% brand_popular]<-
  "Brand_top5"

brand_popular <- tmp$Var1[tmp$Freq>=483 & tmp$Freq<3015]
devicegender$phonebrand_group1[devicegender$phone_brand %in% brand_popular]<-
  "Brand_top10"

brand_popular <- tmp$Var1[tmp$Freq>40 & tmp$Freq<483]
devicegender$phonebrand_group1[devicegender$phone_brand %in% brand_popular]<-
  "Brand_top25"

brand_popular <- tmp$Var1[tmp$Freq>10 & tmp$Freq<=40]
devicegender$phonebrand_group1[devicegender$phone_brand %in% brand_popular]<-
  "Brand_top50"

brand_popular <- tmp$Var1[tmp$Freq<=10]
devicegender$phonebrand_group1[devicegender$phone_brand %in% brand_popular]<-
  "Brand_rare"

# Feature2: Foreign brand/local brand
phone_brand_foreign <- c("三星","谷歌","摩托罗拉","诺基亚","LG",
                         "HCT","西门子","索尼")
devicegender$phonebrand_foreign <- F
devicegender$phonebrand_foreign[devicegender$phone_brand %in% 
                                  phone_brand_foreign]<-T

# Feature3: Old brand/new brand
phone_brand_old <- c("三星","摩托罗拉","诺基亚","LG","HCT","西门子",
                     "华为","TCL")
devicegender$phonebrand_old <- F
devicegender$phonebrand_old[devicegender$phone_brand %in% 
                              phone_brand_old]<-T

write.csv(file="device_genderage.csv",devicegender,row.names = F)

# This is for use in python (neural-network model)
con<-file('device_genderage_utf8.csv',encoding="UTF-8")
write.csv(file=con, devicegender, row.names = F)


## ------------------3.Construct input for XGB model---------------------------
## ----------------------------------------------------------------------------
device_gender <- fread("device_genderage.csv")
device_apps <- fread("device_apps_installed.csv")
device_labels <- fread("device_labels_installed.csv")

d1 <- device_gender[,list(device_id,phone_brand)]
device_gender$phone_brand <- NULL
d2 <- device_gender[,list(device_id,device_model_new)]
device_gender$device_model_new <- NULL
d3 <- device_gender[,list(device_id,phonebrand_group1)]
device_gender$phonebrand_group1 <- NULL
d4 <- device_gender[,list(device_id,phonebrand_foreign)]
device_gender$phonebrand_foreign <- NULL
d5 <- device_gender[,list(device_id,phonebrand_old)]
device_gender$phonebrand_old <- NULL
d6 <- device_apps
d7 <- device_labels

d1[,phone_brand:=paste0("phone_brand:",phone_brand)]
d2[,device_model_new:=paste0("device_model_new:",device_model_new)]
d3[,phonebrand_group1:=paste0("phonebrand_group1:",phonebrand_group1)]
d4[,phonebrand_foreign:=paste0("phonebrand_foreign:",phonebrand_foreign)]
d5[,phonebrand_old:=paste0("phonebrand_old:",phonebrand_old)]
d6[,app_id:=paste0("app_id:",app_id)]
d7[,label_id:=paste0("label_id:",label_id)]

names(d1)<-names(d2)<-names(d3)<-names(d4)<-names(d5)<-
  names(d6)<-names(d7)<-
  c("device_id","feature_name")
dd <- rbind(d1,d2,d3,d4,d5,d6,d7)
rm(d1,d2,d3,d4,d5,d6,d7)
gc()

# create a sparse matrix with all features
ii <- unique(dd$device_id)
jj <- unique(dd$feature_name)
id_i <- match(dd$device_id,ii)
id_j <- match(dd$feature_name,jj)
id_ij <- cbind(id_i,id_j)
M <- Matrix(0,nrow=length(ii),ncol=length(jj),
            dimnames=list(ii,jj),sparse=T)
M[id_ij] <- 1
rm(ii,jj,id_i,id_j,id_ij,dd)
gc()

x <- M[rownames(M) %in% device_gender$device_id,]
id <- device_gender$device_id[match(rownames(x), device_gender$device_id)]
y <- device_gender$group[match(rownames(x),device_gender$device_id)]

# level reduction
x_train <- x[!is.na(y),]
tmp_cnt_train <- colSums(x_train)
x_all <- x[,tmp_cnt_train>0]
rm(x_train,tmp_cnt_train)

# Preparing input for xgboost
group_name <- na.omit(unique(y))

idx_train <- which(!is.na(y))
idx_test <- which(is.na(y))
train_data <- x_all[idx_train,]
test_data <- x_all[idx_test,]
train_label <- match(y[idx_train],group_name)-1
test_label<- match(y[idx_test],group_name)-1
dtrain <- xgb.DMatrix(train_data,label=train_label,missing=NA)
dtest <- xgb.DMatrix(test_data,label=test_label,missing=NA)

## ------------------4. Model training ----------------------------------------
## ----------------------------------------------------------------------------
# Model1: xgblinear
param <- list(booster="gblinear",
              num_class=length(group_name),
              objective="multi:softprob",
              eval_metric="mlogloss",
              eta=0.01,
              lambda=5,
              lambda_bias=0,
              alpha=2)

watchlist <- list(train=dtrain)

# cross-validation
set.seed(314)
fit_cv <- xgb.cv(params=param,
                 data=dtrain,
                 nrounds=1000,
                 watchlist=watchlist,
                 nfold=5,
                 early.stop.round=10,
                 verbose=1)

# model training
ntree <- 212
set.seed(314)
fit_xgblinear <- xgb.train(params=param,
                           data=dtrain,
                           nrounds=ntree,
                           watchlist=watchlist,
                           verbose=1)

# Model2: xgbtree
param <- list(booster="gbtree",
              num_class=12,
              objective="multi:softprob",
              eval_metric="mlogloss",
              eta=0.05,
              max_depth = 5,
              gamma = 4,
              subsample = 0.7,
              colsample_bytree = 0.7,
              min_child_weight = 7)

# cross-validation
set.seed(8548)
fit_cv <- xgb.cv(params=param,
                 data=dtrain,
                 nrounds=3000,
                 watchlist=watchlist,
                 nfold=5,
                 early.stop.round = 10,
                 verbose=1)


# model training
ntree <- 1158
set.seed(8548)
fit_xgbtree <- xgb.train(params=param,
                         data=dtrain,
                         nrounds=ntree,
                         watchlist=watchlist,
                         verbose=1)


## ------------------5. Prediction---------------------------------------------
## ----------------------------------------------------------------------------
pred <- predict(fit_xgblinear,dtest)
pred_detail <- t(matrix(pred,nrow=length(group_name)))
res_submit <- cbind(id=id[idx_test],as.data.frame(pred_detail))
colnames(res_submit) <- c("device_id",group_name)

write.csv(file="submission.csv",res_submit, row.names = F)



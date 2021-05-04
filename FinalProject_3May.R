rm(list=ls())
cat("\014")
print(getwd())
getwd()
setwd("/Users/blouie/Desktop/Baruch MS/SPRING_2021/STA_9890/PROJECT")
library(tree)
library(glmnet)
library(ISLR)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(readr)
library(tidyverse)
set.seed (1)

pitches <- read_csv("pitches.csv", n_max = 1300) # added 'nrows' argument for speed

pitches$type = factor(pitches$type)
pitches$code = factor(pitches$code)
pitches$pitch_type = factor(pitches$pitch_type)

atbats <- read_csv("atbats.csv")
atbats$event = factor(atbats$event)
atbats$p_throws = factor(atbats$p_throws)
atbats$stand = factor(atbats$stand)
atbats$top = factor(atbats$top)

# Joining pitches dataframe with atbats dataframe
# g_id column identifies the season for each game, first 4-digits are the year 
pitches_wgame <- merge(pitches, atbats, by.x = "ab_id", by.y = "ab_id")
head(pitches_wgame)
# Remove id_columns - numerical values w/ no significance
# Remove y0 because it is fixed value
# Remove x, y, could not find clear definition for these values
remove <- c("x","y","y0","g_id", "pitcher_id","batter_id","ab_id")
pitches_wgame <- pitches_wgame[, !colnames(pitches_wgame) %in% remove]

## Remove samples with no data in response variable
pitches_wgame <- pitches_wgame %>% filter(!is.na(end_speed))

# setting up designed matrix
X = model.matrix(end_speed~., pitches_wgame)  #transforms any qualitative variables into dummy variables
y = model.matrix(~end_speed, pitches_wgame)[,2]

n       =   nrow(X)
p       =   ncol(X)
n.train =   floor(0.8*n)
n.test  =   n-n.train


M = 100 # doing for 100 times 

Rsq.test.ls     =     rep(0,M)  # ls = lasso
Rsq.train.ls    =     rep(0,M)
Rsq.test.en     =     rep(0,M)  #en = elastic net
Rsq.train.en    =     rep(0,M)
Rsq.test.rid    =     rep(0,M)  #rid = ridge
Rsq.train.rid   =     rep(0,M)
Rsq.test.rf     =     rep(0,M)  #rf = rondam forest
Rsq.train.rf    =     rep(0,M)


for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit lasso and calculate and record the train and test R squares 
  a=1 # lasso
  cv.fit.las           =     cv.glmnet(X.train, y.train, intercept = T, alpha = a, nfolds = 10)
  fit                  =     glmnet(X.train, y.train,intercept = T, alpha = a, lambda = cv.fit.las$lambda.min)
  y.train.hat          =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat           =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.ls[m]       =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.ls[m]      =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)   
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net 0<a<1
  cv.fit.en            =     cv.glmnet(X.train, y.train, intercept = T, alpha = a, nfolds = 10)
  fit                  =     glmnet(X.train, y.train,intercept = T, alpha = a, lambda = cv.fit.en$lambda.min)
  y.train.hat          =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat           =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]       =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.en[m]      =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  # fit ridge and calculate and record the train and test R squares 
  a=0.0 # ridge
  cv.fit.rid           =     cv.glmnet(X.train, y.train, intercept = T, alpha = a, nfolds = 10)
  fit                  =     glmnet(X.train, y.train,intercept = T, alpha = a, lambda = cv.fit.rid$lambda.min)
  y.train.hat          =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat           =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]      =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rid[m]     =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2) 

  cat(sprintf("m=%3.f| Rsq.test.ls=%.2f,  Rsq.test.en=%.2f,  Rsq.test.rid=%.2f| Rsq.train.ls=%.2f,  Rsq.train.en=%.2f|, Rsq.train.rid=%.2f \n", m,  Rsq.test.ls[m], Rsq.test.en[m], Rsq.test.rid[m], Rsq.train.ls[m], Rsq.train.en[m], Rsq.train.rid[m]))
}

for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
 
  # fit random forrest and calculate and record the train and test R squares 
  
  rf.fit           =     randomForest(X.train,y.train, mtry = floor(sqrt(p)), importance=TRUE)
  y.train.hat      =     predict(rf.fit, newx = X.train) # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(rf.fit, newx = X.test) # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2) / mean((y.test - mean(y.test))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2) / mean((y.train - mean(y.train))^2) 
  
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f | Rsq.train.rf=%.2f | \n", m, Rsq.test.rf[m], Rsq.train.rf[m]))
}


par(mfrow=c(3,1))
plot(cv.fit.las)
plot(cv.fit.en)
plot(cv.fit.rid)

par(mfrow=c(2,4))
boxplot(Rsq.test.ls, main ="lasso-test", col = "orange", border = "brown",notch = TRUE) 
boxplot(Rsq.test.en, main = "elastic net-test", col = "orange", border = "brown",notch = TRUE)  
boxplot(Rsq.test.rid, main = "ridge-test", col = "orange", border = "brown",notch = TRUE) 
boxplot(Rsq.test.rf, main = "random forest-test", col = "orange", border = "brown",notch = TRUE) 

boxplot(Rsq.train.ls, main ="lasso-train",col = "orange", border = "brown",notch = TRUE)
boxplot(Rsq.train.en, main = "elastic net-train",col = "orange", border = "brown",notch = TRUE)
boxplot(Rsq.train.rid, main = "ridge-train",col = "orange", border = "brown",notch = TRUE)
boxplot(Rsq.train.rf, main = "random forest-train" , col = "orange", border = "brown",notch = TRUE) 

# ----------------------------------------------------------------------------------------------------
bootstrapSamples  =     100
beta.ls.bs        =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs        =     matrix(0, nrow = p, ncol = bootstrapSamples)       
beta.rid.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.rf.bs        =     matrix(0, nrow = p, ncol = bootstrapSamples)  



for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs en
  a                =     0.5 
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit.en           =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit.en$beta)
  
  # fit bs lasso
  a                =     1 
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit.ls           =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.ls.bs[,m]   =     as.vector(fit.ls$beta)
  
  # fit boostraP rid
  a                 =     0 
  cv.fit            =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit.rid           =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.rid.bs[,m]   =     as.vector(fit.rid$beta)
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}

# need to change the bootstrapped standard errors calculaton to the upper and lower bounds
ls.bs.sd    = apply(beta.ls.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
rid.bs.sd   = apply(beta.rid.bs, 1, "sd")
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")

#--------------------------------------------------------------------------------------------------------------------------------


# fit en to the whole data
a=0.5 # elastic-net
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.en           =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

# fit lasso to the whole data
a=1 # lasso
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.ls           =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

# fit rid to the whole data
a=0 # lasso
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.rid          =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

# fit random forest to the whole data
rf.fit           =     randomForest(X, y, mtry = floor(sqrt(p)),intercept = FALSE, importance=TRUE)
beta.rf.bs[,m]   =     as.vector(rf.fit$beta)

importance(rf.fit)
varImpPlot(rf.fit)

betaS.en             =     data.frame(c(1:p), as.vector(fit.en$beta)) #6:52 SPH_7_ElNet_lasso_ridge_barplots : no need for error bars
colnames(betaS.en)   =     c( "feature", "value")
betaS.en$feature     =     rownames(fit.en$beta)
betaS.en             <- betaS.en %>% mutate(pos = value > 0)

betaS.ls             =     data.frame(c(1:p), as.vector(fit.ls$beta))
colnames(betaS.ls)   =     c( "feature", "value")
betaS.ls$feature     =     rownames(fit.ls$beta)
betaS.ls             <- betaS.ls %>% mutate(pos = value > 0)

betaS.rid             =     data.frame(c(1:p), as.vector(fit.rid$beta))
colnames(betaS.rid)   =     c( "feature", "value")
betaS.rid$feature     =     rownames(fit.rid$beta)
betaS.rid              <- betaS.rid %>% mutate(pos = value > 0)


lsPlot <-  ggplot(betaS.ls, aes(x=feature, y=value, fill=pos)) +
  geom_bar(stat = "identity", colour="black") +
  ggtitle("lasso") +
  ylab("coefficents") +
  theme(axis.title.x = element_blank()) + 
  theme(plot.title = element_text(hjust=0.5)) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
  scale_fill_manual(values = c("#CC6666", "#66CC99"), guide=FALSE)

enPlot <- ggplot(betaS.en, aes(x=feature, y=value, fill=pos)) +
  geom_bar(stat = "identity", colour="black")  +
  ggtitle("elastic net") +
  ylab("coefficents") +
  theme(axis.title.x = element_blank()) + 
  theme(plot.title = element_text(hjust=0.5)) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
  scale_x_discrete(labels=abbreviate) +
  scale_fill_manual(values = c("#CC6666", "#66CC99"), guide=FALSE)

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value, fill=pos)) +
  geom_bar(stat = "identity", colour="black")  +
  ggtitle("ridge") +
  ylab("coefficents") + 
  theme(plot.title = element_text(hjust=0.5)) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
  scale_x_discrete(labels=abbreviate) +
  scale_fill_manual(values = c("#CC6666", "#66CC99"), guide=FALSE)

rfPlot = varImpPlot(rf.fit)

grid.arrange(lsPlot, enPlot, ridPlot, nrow = 3)

# change the order of factor levels by specifying the order explicitly.
betaS.ls$feature      =  factor(betaS.ls$feature, levels = betaS.ls$feature[order(betaS.ls$value, decreasing = TRUE)])
betaS.en$feature      =  factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.rid$feature     =  factor(betaS.rid$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])

lsPlot <-  ggplot(betaS.ls, aes(x=feature, y=value, fill=pos)) +
  geom_bar(stat = "identity", colour="black") +
  ggtitle("lasso") +
  ylab("coefficents") +
  theme(axis.title.x = element_blank()) + 
  theme(plot.title = element_text(hjust=0.5)) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
  scale_x_discrete(labels=abbreviate) +
  scale_fill_manual(values = c("#CC6666", "#66CC99"), guide=FALSE) +
  ylim(-1, 10) +
  theme(panel.background=element_rect(fill="#F0F0F0"))

enPlot <- ggplot(betaS.en, aes(x=feature, y=value, fill=pos)) +
  geom_bar(stat = "identity", colour="black")  +
  ggtitle("elastic net") +
  ylab("coefficents") +
  theme(axis.title.x = element_blank()) + 
  theme(plot.title = element_text(hjust=0.5)) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
  scale_x_discrete(labels=abbreviate) +
  scale_fill_manual(values = c("#CC6666", "#66CC99"), guide=FALSE) +
  ylim(-1,10) +
  theme(panel.background=element_rect(fill="#F0F0F0"))

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value, fill=pos)) +
  geom_bar(stat = "identity", colour="black")  +
  ggtitle("ridge") +
  ylab("coefficents") + 
  theme(plot.title = element_text(hjust=0.5)) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
  scale_x_discrete(labels=abbreviate) +
  scale_fill_manual(values = c("#CC6666", "#66CC99"), guide=FALSE) +
  theme(panel.background=element_rect(fill="#F0F0F0"))

grid.arrange(lsPlot, enPlot, ridPlot,nrow = 3)


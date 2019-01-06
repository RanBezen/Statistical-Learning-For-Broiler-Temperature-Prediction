.libPaths("C:/R")
library(plotly)
library(car)
require(glmnet)
require(leaps)
require(caret)
require(ggplot2)
require(randomForest)
library(neuralnet)
require(factoextra)



res<-read.csv('dataset.csv', header = TRUE, stringsAsFactors = FALSE)
set.seed(8)
#-------------------------------feature engineering--------------------------------#
res$time_num <- round(sin(res$time_num*pi),digits = 3)
drops <- c("time","date")
res<-res[ , !(names(res) %in% drops)]

data_scale<-function(data){
  maxs <- apply(data, 2, max) 
  mins <- apply(data, 2, min)
  scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
  return(scaled)
}
###########----------Train Test split-----------############

n <- nrow(res)
shuffled_df <- res[sample(n), ]
train_indices <- 1:round(0.65 * n)
train <- shuffled_df[train_indices, ]
test_indices <- (round(0.65 * n) + 1):n
test <- shuffled_df[test_indices, ]

#--------------------- lasso regression ---------------------#
x <- as.matrix(data_scale(train[,-ncol(train)])) # Removes class
y <- as.double(as.matrix(train[,ncol(train)])) # Only class
alpha<-0.5
cv.lasso <- cv.glmnet(x, y, family='gaussian', alpha=alpha,nfolds =10)

# Results
plot(cv.lasso,type.coef="2norm")
plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
opt_lambda<-cv.lasso$lambda.min
lambda.1se<-cv.lasso$lambda.1se
coef(cv.lasso, s=cv.lasso$lambda.min)
R_Squared =  1 - cv.lasso$cvm/var(y)
plot(cv.lasso$lambda,R_Squared)

lasso_fitted <- glmnet(x=x, y=y, alpha = alpha, lambda=opt_lambda)
k<-length(lasso_fitted$beta@i)-1
k
r2 <- cv.lasso$glmnet.fit$dev.ratio[which(cv.lasso$glmnet.fit$lambda == cv.lasso$lambda.min)]
r2
r_adj <- 1-((1-r2)*(nrow(train)-1)/(nrow(train)-1-k))
r_adj

y_predicted <- predict(lasso_fitted, newx = x)
shapiro.test(y_predicted)
residuals <- y_predicted - y
q2<-plot(y_predicted, residuals, 
     ylab="Residuals", xlab="temp") 
abline(0, 0)    

q1<-qqPlot(y_predicted)

#BIC
BIC(lasso_fitted)

tLL <- lasso_fitted$nulldev - deviance(lasso_fitted)
k_lasso <- lasso_fitted$df
n_lasso <- lasso_fitted$nobs

BIC<-log(n_lasso)*k_lasso - tLL
BIC
#---------------------- linear backward regression -------------------#

# Set up repeated k-fold cross-validation
train.control <- trainControl(method = "LOOCV")
# Train the model
data<- train
step.model <- train(y_temp ~., data = data,
                    method = "leapBackward", 
                    tuneGrid = data.frame(nvmax = 1:ncol(data)),
                    trControl = train.control
)
step.model$results
step.model$bestTune
summary(step.model$finalModel)
final<-coef(step.model$finalModel, step.model$bestTune[[1]])
bckwrd_names<-c(names(final)[-1], 'y_temp')
bckwrd_names
#data<-train[,bckwrd_names]
bckwrd_fitted<-lm(y_temp ~., data=train[,bckwrd_names])
BIC(bckwrd_fitted)
anova(bckwrd_fitted)
summary(bckwrd_fitted)
shapiro.test(predict(bckwrd_fitted, newdata = train[,bckwrd_names]))

qqPlot(bckwrd_fitted)

eruption.res = bckwrd_fitted$residuals
plot(bckwrd_fitted$fitted.values, eruption.res, 
     ylab="Residuals", xlab="temp") 
abline(0, 0)    

par(mar = rep(2, 4))
opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(bckwrd_fitted,las = 1)
par(opar)


########################-------------------PcA
#create pca dataset by training data
PCA_Selection<-function(dataset){
  require(factoextra)
  dataset.label <- dataset[, ncol(dataset)]
  dataset<-dataset[,-ncol(dataset)]
  dataset.pca <- prcomp(dataset, scale = TRUE)
  
  pca_pc<-dataset.pca$x
  summary(dataset.pca)
  
  fviz_eig(dataset.pca)
  par(mar=c(4,4,2,2))
  fviz_pca_var(dataset.pca, col.var="steelblue")+
    theme_minimal()
  
  #create a new dataset after the PCA
  q<-summary(dataset.pca)$importance
  i<-1
  nPC<-0
  for(i in 1:ncol(q)){
    if(q[3,i]<0.8){
      nPC<-i
    }
  }
  
  nPC<-nPC+1
  
  After_pca_data <- data.frame(dataset.pca$x[,1:nPC])
  After_pca_data$y_temp<-dataset.label
  return(After_pca_data)
}
#create pca test dataset by training data
PCA_Pred<-function(train,test){
  require(factoextra)
  dataset<-train[,-ncol(train)]
  dataset.pca <- prcomp(dataset, scale = TRUE)
  
  q<-summary(dataset.pca)$importance
  i<-1
  nPC<-0
  for(i in 1:ncol(q)){
    if(q[3,i]<0.8){
      nPC<-i
    }
  }
  
  nPC<-nPC+1
    
  pred <- predict(dataset.pca, newdata=test[,-ncol(test)])
  pred <-  data.frame(pred[,1:nPC])
  pred$y_temp<-test[,ncol(test)]
  
  return(pred)
}
#-------------------PCA regression-------------------#

dataPCA<-PCA_Selection(train)

pca_fitted<-lm(y_temp ~., data=dataPCA)
BIC(pca_fitted)

anova(pca_fitted)
summary(pca_fitted)

shapiro.test(predict(pca_fitted,dataPCA))

qqPlot(pca_fitted)

eruption.res = pca_fitted$residuals
plot(pca_fitted$fitted.values, eruption.res, 
     ylab="Residuals", xlab="temp") 
abline(0, 0)    

opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(pca_fitted,las = 1)
par(opar)

#####___________random forest_________________######

randomForest_cv_tuning<-function(data){
  require(ggplot2)
  require(randomForest)
  #data<-train
  treesLimit<-50
  numtree <- seq(1,treesLimit,3)
  numheight<-seq(2,ncol(data)-1,1)
  MSE<- matrix(0,length(numtree)*length(numheight),1)
  numtrees_rf<-MSE
  numheights_rf<-MSE
  kfoldsVec<- matrix(0,10,1)
  i<- 1
  
  kfolds <- cut(seq(1,nrow(data)),breaks=10,labels=FALSE)
  counter=1
  for(height in numheight){
    for(trees in numtree){
      for(l in 1:10){
        p<-(paste('step: ',(counter),' from: ',10*length(numtree)*length(numheight)))
        cat(p, "\n")
        testIndexes<- which(kfolds==l,arr.ind=TRUE)
        val<- data[testIndexes, ]
        train<- data[-testIndexes, ]
        
        rf<- randomForest(y_temp ~., data=train, ntree=trees, mtry=height)
        pred<-predict(rf,val[,-ncol(val)]) #Predictions on Test Set for each Tree
        kfoldsVec[l]= with(val[,-ncol(val)], mean( (val[,ncol(val)] - pred)^2)) #Mean Squared Test Error
        counter<-counter+1
      }
      MSE[i] <- mean(kfoldsVec)
      numtrees_rf[i] <- trees
      numheights_rf[i] <- height
      i <- i+1
    }
  }
  
  #validation MSE
  minMSE<-min(MSE)
  minMSE
  
  #the optimal number of trees
  ntrees        <- numtrees_rf[which.min(MSE)]
  ntrees
  
  #the height of the trees
  nheight     <-numheights_rf[which.min(MSE)]
  nheight
  result<-list(ntrees,nheight,numtrees_rf,numheights_rf,MSE)
  return(result)
}
#testing the best configuration
results_rf<-randomForest_cv_tuning(train)

ntrees <-results_rf[[1]]
ntrees
nheight <-results_rf[[2]]
nheight
numtree <-results_rf[[3]]
numtree
numheight <-results_rf[[4]]
MSE <-results_rf[[5]]
min(MSE)
plot(numtree[which(numheight == nheight)],MSE[which(numheight == nheight)])
plot(numheight[which(numtree == ntrees)],MSE[which(numtree == ntrees)])

rm(result_Matrix)
result_Matrix<-data.frame(numtree,numheight,MSE)
write.csv(result_Matrix,'result_Matrix_rf.csv')


#rf.fit<- randomForest(y_temp ~., data=train, ntree=43, mtry=7)
rf.fit<- randomForest(y_temp ~., data=train, ntree=ntrees, mtry=nheight)
rf.fit
which.min(rf.fit$mse)
rf.fit$importance
plot(rf.fit)
plot(rf.fit$mse)


#____________________________neural net 3 hidden layers____________________________


neuralNet_3hidden_cv_tuning<-function(data){
  data<-train
  library(neuralnet)
  
  maxs <- apply(data, 2, max) 
  mins <- apply(data, 2, min)
  scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
  
  nnum1 <- seq(3,5,1)
  nnum2 <- seq(3,5,1)
  nnum3 <- seq(3,4,1)  
  i    <- 1 
  kfolds <- cut(seq(1,nrow(data)),breaks=10,labels=FALSE)
  counter_nn<-1
  kfoldsVec<- matrix(0,10,1)
  MSE_nn  <- matrix(0,length(nnum1)*length(nnum2)*length(nnum3),1)
  n1<-MSE_nn
  n2<-MSE_nn
  n3<-MSE_nn
  for(neurons3 in nnum3){
    for(neurons2 in nnum2){
      for(neurons1 in nnum1){
        for(l in 1:10){
          p<-(paste('-----------step: ',(counter_nn),' from: ',10*length(nnum1)*length(nnum2)*length(nnum3),'----------'))
          cat(p, "\n")
          p1<-(paste('neurons: ',neurons1,':',neurons2,':',neurons3))
          cat(p1, "\n")
          
          testIndexes<- which(kfolds==l,arr.ind=TRUE)
          val<- scaled[testIndexes, ]
          trainCV<- scaled[-testIndexes, ]
          n <- names(trainCV)
          f <- as.formula(paste("y_temp ~", paste(n[!n %in% "y_temp"], collapse = " + ")))
          nn <- neuralnet(f,data=trainCV,hidden=c(neurons1,neurons2,neurons3),linear.output=T)
          
          pr.nn <- compute(nn,val[,-ncol(val)])
          pr.nn_ <- pr.nn$net.result*(max(data$y_temp)-min(data$y_temp))+min(data$y_temp)
          pr.nn_<-round(pr.nn_,digits = 2)
          test.r <- (val$y_temp)*(max(data$y_temp)-min(data$y_temp))+min(data$y_temp)
          kfoldsVec[l]=sum((test.r - pr.nn_)^2)/nrow(val)
          p1<-(paste('cv',l,'| MSE: ',sum((test.r - pr.nn_)^2)/nrow(val)))
          cat(p1, "\n")
          counter_nn<-counter_nn+1
        }
        MSE_nn[i]         <- mean(kfoldsVec)
        n1[i]<-neurons1
        n2[i]<-neurons2
        n3[i]<-neurons3
        p1<-(paste('__________________________________mean MSE: ',mean(kfoldsVec),'________________________________'))
        cat(p1, "\n")
        i              <- i + 1
      }
    }
  }
  minMSE<-min(MSE_nn)
  minMSE
  #optimal number of neurons
  neurons_1       <- n1[which.min(MSE_nn)]
  neurons_1
  neurons_2       <- n2[which.min(MSE_nn)]
  neurons_2
  neurons_3       <- n3[which.min(MSE_nn)]
  neurons_3
  result<-list(neurons_1,neurons_2,neurons_3,n1,n2,n3,MSE_nn)
  return(result)
}
results_nn<-neuralNet_3hidden_cv_tuning(train)
neurons_1<-results_nn[[1]]
neurons_1 
neurons_2<-results_nn[[2]]
neurons_2 
neurons_3<-results_nn[[3]]
neurons_3 
n1<-results_nn[[4]]
n2<-results_nn[[5]]
n3<-results_nn[[6]]
MSE_nn<-results_nn[[7]]
#n1=5 n2=5 n3=3
plot(n1[which(n2[which(n3 == neurons_3)] == neurons_2)],MSE_nn[which(n2[which(n3 == neurons_3)] == neurons_2)], main = 'MSE vs layer 1 (layers 2,3 const.)',xlab = 'neurons',ylab = 'MSE')
plot(n2[which(n1[which(n3 == neurons_3)] == neurons_1)],MSE_nn[which(n1[which(n3 == neurons_3)] == neurons_1)], main = 'MSE vs layer 2 (layers 1,3 const.)',xlab = 'neurons',ylab = 'MSE')
plot(n3[which(n1[which(n2 == neurons_2)] == neurons_1)],MSE_nn[which(n1[which(n2 == neurons_2)] == neurons_1)], main = 'MSE vs layer 3 (layers 1,2 const.)',xlab = 'neurons',ylab = 'MSE')

min(MSE_nn)
rm(result_Matrix)
result_Matrix<-data.frame(n1,n2,n3,MSE_nn)
write.csv(result_Matrix,'result_Matrix_nn3.csv')
n <- names(train)
f <- as.formula(paste("y_temp ~", paste(n[!n %in% "y_temp"], collapse = " + ")))
nn.fit <- neuralnet(f,data=data_scale(train),hidden=c(neurons_1,neurons_2,neurons_3),linear.output=T)
#nn.fit <- neuralnet(f,data=data_scale(train),hidden=c(5,5,3),linear.output=T)

plot(nn.fit)



#-------------------------------------------------------test-----------------------------------#

lasso_data <- as.matrix(data_scale(test[,-ncol(test)]))
y_pred_lasso <- predict(lasso_fitted, newx = lasso_data)

bckwrd_data<-test[,bckwrd_names]
y_pred_bckwrd <- predict(bckwrd_fitted, newdata = bckwrd_data[,-ncol(bckwrd_data)])

pca_data<-PCA_Pred(train,test)
y_pred_pca <- predict(pca_fitted, newdata = pca_data[,-ncol(pca_data)])

y_pred_rf <- predict(rf.fit,test[,-ncol(test)])

nn_dataM<-data_scale(test)
y_pred_nn_unscalse <- compute(nn.fit,nn_dataM[,-ncol(nn_dataM)])
y_pred_nn <- y_pred_nn_unscalse$net.result*(max(train$y_temp)-min(train$y_temp))+min(train$y_temp)

y_preds<-cbind(y_pred_lasso,y_pred_bckwrd,y_pred_pca,y_pred_rf,y_pred_nn)
y_preds<-round(y_preds,digits = 2)
y<-test[,ncol(test)]

models<-list(lasso_fitted,bckwrd_fitted,pca_fitted,rf.fit,nn.fit)
models_names<-c('lasso','backward','PCA','rf','nn')
results<- matrix(0,5,3)

for (i in 1:ncol(y_preds)){
  
  sst <- sum((y - mean(y_preds[,i]))^2)
  sse <- sum((y_preds[,i] - y)^2)
  r2 <- 1 - sse / sst
  RMSE <- RMSE(y,y_preds[,i])
  MAPE<-sum(abs((y-y_preds[,i])/y_preds[,i]))/length(y)*100
  results[i,1]<-r2
  results[i,2]<-RMSE
  results[i,3]<-sum(abs(y-y_preds[,i])/y_preds[,i])
  
}
results <-data.frame(results)
colnames(results)<-c('Rsquare','RMSE','MAPE')
rownames(results)<-models_names
results
plot(results[,1])

x<-seq(1,5,1)
plot(x,results[,1], type="b", col="green", lwd=3, pch=15,
     xlab="model", ylab="score", ylim=range(results[,2],results[,1]))
lines(x,results[,2], type="b", col="red", lwd=3, pch=15)

title("models scores Rsqr RMSE")

legend("right",c("Rsqr","RMSE"), lwd=c(5,5), col=c("green","red"),  lty=1:2, cex=0.5, inset=.02)
legend("left",c("lasso","backward","PCA","RF","NN"), lwd=c(5,5), col=c("green","red"),  lty=1:2, cex=0.5, inset=.02)

plot(x,results[,3], type="b", col="blue", lwd=3, pch=15,
     xlab="model", ylab="score", ylim=range(results[,3]))
title("models scores MAPE")

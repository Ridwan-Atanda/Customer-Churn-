##Set Working Directory
setwd("/Users/atandaridwan/Desktop/Machine learning ")

##Loading the data
bank = read.csv("/Users/atandaridwan/Desktop/Machine learning /default of credit card clients.csv", header = T, sep = ",")

##Cleaning
#Checking for missing values
sum(is.na(bank))
str(bank)


#Removing irrelevant columns
bank$ID = NULL
bank$LIMIT_BAL = NULL
bank$PAY_0 = NULL
bank$PAY_2 = NULL
bank$PAY_3 = NULL
bank$PAY_4 = NULL
bank$PAY_5 = NULL
bank$PAY_6 = NULL

#checking the strings of all columns and encoding them to the right strings
str(bank)


#Encoding all categorical variables
bank$SEX = factor(bank$SEX, 
                         labels = c("Female","Male"), 
                         levels = c(2,1))

bank$EDUCATION[bank$EDUCATION == 5] = 4
bank$EDUCATION[bank$EDUCATION == 6] = 4
bank$EDUCATION[bank$EDUCATION == 0] = 4
bank$EDUCATION = factor(bank$EDUCATION, 
                  labels = c("graduate school","university","high school","other"), 
                  levels = c(1,2,3,4))

bank$MARRIAGE = as.factor(bank$MARRIAGE)
bank$MARRIAGE[bank$MARRIAGE == 0] = 3
bank$MARRIAGE = factor(bank$MARRIAGE, 
                  labels = c("married","single","others"), 
                  levels = c(1,2,3))

bank$default.payment.next.month = factor(bank$default.payment.next.month, 
                  labels = c("No","Yes"), 
                  levels = c(0,1))


#Descriptive statistics
summary(bank)
table(bank$default.payment.next.month)
prop.table(table(bank$default.payment.next.month)) #with the percentage, it is not balanced.
colors = c("green", "orange")
barplot(table(bank$default.payment.next.month), main = "Distribution of Defaulters", ylab="Frequency", col = colors)

#Setting train and test data
library(caret)
library(ggplot2)
library("lattice")
set.seed(2331)
sample = createDataPartition(bank$default.payment.next.month, p = .75, list = FALSE) 
train = bank[sample, ]
test = bank[-sample, ]

#Performing Synthetic Minority Over-sampling Technique (SMOTE) to minimize the dataset class imbalance.
library(DMwR)
SMOTEtrain = SMOTE(default.payment.next.month ~ ., train, perc.over = 100, k = 5, perc.under = 200)
table(SMOTEtrain$default.payment.next.month)
barplot(table(SMOTEtrain$default.payment.next.month), main = "Distribution of SMOTE Defaulters", ylab="Frequency", col = colors)



#NAIVE BAYES
#Building a predictive model (Naive Bayes) with Synthetic Minority Over-sampling Technique (SMOTE)
library(naivebayes)
SMOTEnbmodel = naive_bayes(default.payment.next.month~., data = SMOTEtrain, usekernel = TRUE)
SMOTEnbmodel

#Predictive Model Evaluation with SMOTEtrain data
prediction = predict(SMOTEnbmodel, SMOTEtrain, type = 'prob')
prediction1 = predict(SMOTEnbmodel, SMOTEtrain)
(NBTable = table(prediction1, SMOTEtrain$default.payment.next.month))
sum(diag(NBTable)) / sum(NBTable)
1 - sum(diag(NBTable)) / sum(NBTable)
head(cbind(prediction1, SMOTEtrain))

#ConfusionMatrix for test data
prediction2 = predict(SMOTEnbmodel, test)
(table2 = table(prediction2, test$default.payment.next.month))
sum(diag(table2)) / sum(table2)
1 - sum(diag(table2)) / sum(table2)
head(cbind(prediction2, test))
confusionMatrix(prediction2, test$default.payment.next.month, positive = "Yes")
plot(SMOTEnbmodel)

#LOGISTIC REGRESSION
#Building a predictive model (Logistic Regression) with Synthetic Minority Over-sampling Technique (SMOTE)
library(glmnet)
SMlogRegModel = glm(SMOTEtrain$default.payment.next.month ~., data = SMOTEtrain, family='binomial')
summary(SMlogRegModel)

#Predictive Model Evaluation with SMOTEtrain data
LRprediction = predict(SMlogRegModel, SMOTEtrain, type = 'response')
head(LRprediction)

LRprediction1 = ifelse(LRprediction>0.5, 1, 0)
LRTable = table(Predicted = LRprediction1, Actual = SMOTEtrain$default.payment.next.month)
LRTable
sum(diag(LRTable)) / sum(LRTable)
1 - sum(diag(NBTable)) / sum(NBTable)
head(cbind(LRprediction1, SMOTEtrain))

#Predictive Model Evaluation with test data
Test_prediction1 = predict(SMlogRegModel, test, type = 'response')
head(Test_prediction1)

LRprediction2 = ifelse(Test_prediction1>0.5, 1, 0)
LRtable2 = table(Predicted = LRprediction2, Actual = test$default.payment.next.month)
LRtable2
sum(diag(LRtable2)) / sum(LRtable2)
1 - sum(diag(LRtable2)) / sum(LRtable2)
head(cbind(LRprediction2, test))




#Setting Working Directory
setwd("/Users/atandaridwan/Desktop/Machine learning ")

#Loading the data
Bank_churn = read.csv("/Users/atandaridwan/Desktop/DATA MINING & MACHINE LEARNING /Machine learning with R/Churn_Modelling.csv", header = TRUE, sep = ",")

#Checking for missing values
sum(is.na(Bank_churn))
colSums(is.na(Bank_churn))


#checking the strings of all columns and encoding them to the right strings
str(Bank_churn)

#summary statistics
summary(Bank_churn)

#Conversion of variables to appropriate strings
Bank_churn$NumOfProducts = as.factor(Bank_churn$NumOfProducts)
Bank_churn$HasCrCard = as.factor(Bank_churn$HasCrCard)
Bank_churn$IsActiveMember = as.factor(Bank_churn$IsActiveMember)
Bank_churn$Exited = as.numeric(Bank_churn$Exited)

#Removing irrelevant columns
Bank_churn$RowNumber = NULL
Bank_churn$CustomerId = NULL
Bank_churn$Surname = NULL

#Correlation test for relationship
cor.test(Bank_churn$Exited, Bank_churn$CreditScore)
cor.test(Bank_churn$Exited, Bank_churn$Age)
cor.test(Bank_churn$Exited, Bank_churn$Tenure)
cor.test(Bank_churn$Exited, Bank_churn$Balance)
cor.test(Bank_churn$Exited, Bank_churn$EstimatedSalary)


#Encoding all categorical variables
Bank_churn$NumOfProducts= factor(Bank_churn$NumOfProducts, 
                              labels = c("Pdt A","Pdt B","Pdt C","Pdt D"), 
                              levels = c(1,2,3,4))
Bank_churn$HasCrCard = factor(Bank_churn$HasCrCard, 
                              labels = c("No","Yes"), 
                              levels = c(0,1))
Bank_churn$IsActiveMember = factor(Bank_churn$IsActiveMember, 
                              labels = c("No","Yes"), 
                              levels = c(0,1))
Bank_churn$Exited = factor(Bank_churn$Exited, 
                              labels = c("No","Yes"), 
                              levels = c(0,1))


#Distribution of relevant columns
colors = c("green", "orange")
barplot(table(Bank_churn$Exited), main = "Distribution of Exited", ylab="Frequency", col = colors)

colors = "blue"
hist(Bank_churn$Age, main = "Distribution of Age", ylab="Histogram", col = colors)

colors = "red"
hist(Bank_churn$Balance, main = "Distribution of Balance", ylab="Histogram", col = colors)

plot(Bank_churn$Exited, Bank_churn$Age, ylab="Age", xlab="Exited")
plot(Bank_churn$Exited, Bank_churn$Balance, ylab="Balance", xlab="Exited")

plot(Bank_churn$NumOfProducts,main = "Distribution of Number of Products", ylab="Frequency", col = colors)
colors = c("green", "orange","blue","red")

plot(Bank_churn$NumOfProducts~Bank_churn$EstimatedSalary, col=colors)


#Descriptive statistics
summary(Bank_churn)

#Setting train and test data
library(caret)
set.seed(4539)
sample = createDataPartition(Bank_churn$Exited, p = .80, list = FALSE) 
train = Bank_churn[sample, ]
test = Bank_churn[-sample, ]

#Performing Synthetic Minority Over-sampling Technique (SMOTE) to minimize the dataset class imbalance.
library(DMwR)
SMOTEtrain = SMOTE(Exited ~ ., train, perc.over = 100, k = 5, perc.under = 200)
table(SMOTEtrain$Exited)
barplot(table(SMOTEtrain$Exited), main = "Distribution of SMOTE Exited", ylab="Frequency", col = colors)



#C5.0 DECISION TREE
#C5.0 DECISION TREE with SMOTE
library(C50)
SMOTEcFiftyModel = C5.0(x =SMOTEtrain[,1:10], y = SMOTEtrain$Exited)
summary(SMOTEcFiftyModel)
plot(SMOTEcFiftyModel)

#Predictions
pred = predict(SMOTEcFiftyModel, test[,1:10])
C50table = table(pred, test$Exited)
C50table
sum(diag(C50table)) / sum(C50table)
1 - sum(diag(C50table)) / sum(C50table)
head(cbind(pred, test))
confusionMatrix(pred, test$Exited, positive = "Yes")


#C5.0 DECISION TREE WITH RULES
#C5.0 DECISION TREE WITH RULES with SMOTE
rulesModel = SMOTEcFiftyModel = C5.0(x =SMOTEtrain[,1:10], y = SMOTEtrain$Exited, rules = TRUE)
summary(rulesModel)

#Predictions
Rules_pred = predict(rulesModel, test[,1:10])
Rule_table = table(Rules_pred, test$Exited)
Rule_table
sum(diag(Rule_table)) / sum(Rule_table)
1 - sum(diag(Rule_table)) / sum(Rule_table)
head(cbind(Rules_pred, test))
confusionMatrix(Rules_pred, test$Exited, positive = "Yes")

#END

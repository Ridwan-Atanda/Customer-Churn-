#Setting Working Directory
setwd("/Users/atandaridwan/Desktop/Machine learning ")

#Loading the data
telchurn = read.csv("/Users/atandaridwan/Desktop/Machine learning /Telco Customer Churn.csv", header = TRUE, sep = ",")

#Checking for missing values and replaced with mean of the column of the missing values
sum(is.na(telchurn))
colSums(is.na(telchurn))
mean = mean(telchurn$TotalCharges, na.rm = TRUE)
telchurn$TotalCharges[is.na(telchurn$TotalCharges)]= '2283.3'
sum(is.na(telchurn))


#Removing irrelevant columns
telchurn$customerID = NULL
telchurn$Partner = NULL
telchurn$PhoneService = NULL
telchurn$InternetService = NULL
telchurn$OnlineSecurity = NULL
telchurn$OnlineBackup = NULL
telchurn$DeviceProtection = NULL
telchurn$StreamingTV = NULL
telchurn$StreamingMovies = NULL
telchurn$PaperlessBilling = NULL

#checking the strings of all columns and encoding them to the right strings
str(telchurn)
summary(telchurn)
telchurn$TotalCharges = as.numeric(telchurn$TotalCharges)
telchurn$SeniorCitizen = as.factor(telchurn$SeniorCitizen)
str(telchurn)

#Encoding all categorical variables
telchurn$gender = factor(telchurn$gender, 
                         levels = c("Female","Male"), 
                         labels = c(0,1))
telchurn$Dependents = factor(telchurn$Dependents,
                             levels = c("No","Yes"), 
                             labels = c(0,1))
telchurn$TechSupport = factor(telchurn$TechSupport,
                              levels = c("No internet service","No","Yes"), 
                              labels = c(1,2,3))
telchurn$MultipleLines = factor(telchurn$MultipleLines,
                                levels = c("No phone service","No","Yes"), 
                                labels = c(1,2,3))
telchurn$Contract = factor(telchurn$Contract,
                           levels = c("Month-to-month","One year","Two year"), 
                           labels = c(1,2,3))
telchurn$PaymentMethod = factor(telchurn$PaymentMethod,
                                levels = c("Bank transfer (automatic)","Credit card (automatic)","Electronic check","Mailed check"), 
                                labels = c(1,2,3,4))
telchurn$Churn = factor(telchurn$Churn,
                        levels = c("No","Yes"), 
                        labels = c(0,1))



#Descriptive statistics
summary(telchurn)
table(telchurn$Churn)
prop.table(table(telchurn$Churn)) #with the percentage, it is not balanced.
colors = c("green", "orange")
barplot(table(telchurn$Churn), main = "Distribution of Churn", ylab="Frequency", col = colors)


#Setting train and test data
library(caret)
set.seed(735637)
sample = createDataPartition(telchurn$Churn, p = .75, list = FALSE) 
train = telchurn[sample, ]
test = telchurn[-sample, ]

#Performing Synthetic Minority Over-sampling Technique (SMOTE) to minimize the dataset class imbalance.
library(DMwR)
SMOTEtrain = SMOTE(Churn ~ ., train, perc.over = 100, k = 5, perc.under = 200)
table(SMOTEtrain$Churn)
barplot(table(SMOTEtrain$Churn), main = "Distribution of SMOTE Churn", ylab="Frequency", col = colors)
barplot(table(test$Churn), main = "Distribution of SMOTE Churn", ylab="Frequency", col = colors)

#RANDOM FOREST
#Predictive Model (Random Forest) with SMOTE
library(randomForest)
SMOTErfModel = randomForest(Churn~., data = SMOTEtrain)
SMOTErfModel
plot(SMOTErfModel)
varImpPlot(SMOTErfModel)
#Predictive Model Evaluation with test data
confusionMatrix(predict(SMOTErfModel, test), test$Churn, positive = '1')
head(confusionMatrix(predict(SMOTErfModel, test), test$Churn, positive = '1'))



#KNN
#Predictive Model (KNN) with SMOTE

#kNN requires normalised data

library(class)
normalize = function(x) { return ((x - min(x)) / (max(x) - min(x))) }
#Normalizing our dataset so that calculating the distances in the feature space makes sense
Churn_n = subset(telchurn, select=c(4,9:10)) #get the numerical for normalisation -- kNN also doesn't support levelled factors either
Churn_n = as.data.frame(lapply(Churn_n, normalize)) #normalise
summary(Churn_n) #all our numericals are normalised, our categoricals are untouched

#re make train and test note we can retain the original distribution if we choose to
train_n = Churn_n[sample, ]
test_n = Churn_n[-sample, ]

#different ways to determine k
k1 = round(sqrt(dim(train_n)[1])) #sqrt of number of instances

knn1 = knn(train = train_n, test = test_n, cl = train$Churn, k=k1)

(knn1Acc = 1- mean(knn1 != test$Churn))


#Setting a benchmark
#That every custpmers stays:
str(test$Churn) # remain (Y) is 0
benchmark = rep(0, dim(test)[1])
(accuracyBM = 1 - mean(benchmark != test$Churn))
library(gmodels)
CrossTable(x = test$Churn, y = knn1,
           prop.chisq = FALSE)



#that customers who are senior citizens are less likely to leave
str(telchurn$SeniorCitizen)
benchmark2 = rep(0, dim(test)[1])
benchmark2[test$SeniorCitizen == 'No'] = 1
benchmark2[test$SeniorCitizen == 'Yes'] = 1
(accuracyBM2 = 1 - mean(benchmark2 != test$SeniorCitizen))


knn1Acc - accuracyBM #same as benchmark
knn1Acc - accuracyBM2 #worse than benchmark


library(plyr)
library(ggplot2)
set.seed(123)

# Create training and testing data sets
idx = sample(1:nrow(telchurn), size = 100)
train.idx = 1:nrow(telchurn) %in% idx
test.idx =  ! 1:nrow(telchurn) %in% idx

train = telchurn[train.idx, 1:4]
test = telchurn[test.idx, 1:4]

# Get labels
labels = telchurn[train.idx, 5]

# Do knn
fit = knn(train, test, labels)
fit

# Create a dataframe to simplify charting
plot.df = data.frame(test, predicted = fit)

# Use ggplot
# 2-D plots example only
# Sepal.Length vs Sepal.Width

# First use Convex hull to determine boundary points of each cluster
plot.df1 = data.frame(x = plot.df$SeniorCitizen, 
                      y = plot.df$tenure, 
                      predicted = plot.df$predicted)

find_hull = function(df) df[chull(df$x, df$y), ]
boundary = ddply(plot.df1, .variables = "predicted", .fun = find_hull)

ggplot(plot.df, aes(SeniorCitizen, plot.df$tenure, color = predicted, fill = predicted)) + 
  geom_point(size = 5) + 
  geom_polygon(data = boundary, aes(x,y), alpha = 0.5)

#END


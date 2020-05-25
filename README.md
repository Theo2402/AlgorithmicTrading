# Credit Card Fraud Detection


We'll use here data from Kaggle. It can be downloaded here: https://www.kaggle.com/mlg-ulb/creditcardfraud

I'll be using R for this little project.
In this dataset we are trying to predict if a fraud was performed (Class = 1) or no (class = 0). Therefore we want to predict the "Class" column.
The authors performed a PCA. That's why the columns labels are in the form of "V + number".
Time is the delay between the different transactions and the first transaction. Time is not relevant to predict a Fraud so we can drop the column.  
```bash
creditcard <- read.csv("/Users/diloretotheo/Downloads/creditcard.csv")
```

--- 
# Data summary


```bash
sum(is.na(creditcard))
summary(creditcard)
```


```bash
library(psych)
multi.hist(creditcard)
```
```bash
hist(creditcard$Class,xlab = "Class",col="blue")  # can see barely class 1 values
```
Checking the "Class" column
```bash
count <- table(creditcard$Class)
count  # 492 values of 1 and 284315 value of 0
```

```bash
c = as.vector(count)
Fraud = c[2]/(c[2]+c[1])*100
Fraud       # 0.172%
```

# Data Normalization:
We need to normalize the column 'Amount'.
To normalize between 0 and 1 using the MaxMin we can use the following equation: (x - min(x)) / (max(x) - min(x)).
Howevern to have it between -1 and 1 we need to multiply by 2 and remove 1:

normalize <- function(x) {
  return (2*((x - min(x)) / (max(x) - min(x)))-1) 
}

Let's first have a look at the data:
```bash
plot(creditcard$Amount) # there are many outliers with high values. 
hist(creditcard$Amount,xlab = "Amount",col="blue")  # all observations are less than 2500
```

We need to remove oultiers. Failing to do so all will get all our normalized values near one extreme.
We can use boxplot to highlight outliers and remove them:

```bash
boxplot(creditcard$Amount)
length(boxplot(creditcard$Amount, plot=FALSE)$out)
```
```bash
out <- boxplot(creditcard$Amount, plot=FALSE)$out   # All outliers.
Data <- creditcard[-which(creditcard$Amount %in% out),]   #removing outliers from our data.
```
```bash
countout <- table(Data$Class)
countout        # 401 
```
We would lose 91 "class 1" values using the MaxMin method. 


We can try another normalization method to avoid losing data. We can use the mean and Standard Deviation
```bash
set.seed(123)
normalizedAmount <- (creditcard$Amount - mean(creditcard$Amount)) / sd(creditcard$Amount)
```
we can use also the scale function:  ```bashnormalizedAmount <- scale(creditcard$Amount)```


We can gather the data and remove non-normalized Data.
```bash
Dataset <- data.frame(creditcard[, 2:29], normalizedAmount,creditcard[,31])
colnames(Dataset)
colnames(Dataset)[30] <- "Class"
colnames(Dataset)
```
The data is extremely unbalanced. The Machine Learning algorithm won't be able to predict with such unbalanced data.
 We have to balance the "Class" column. We can start by getting all the rows for class=1.
 ```bash
Class1 <- Dataset[which(Dataset$Class=='1'), ] 
nrow(Class1)  #492 rows
Class0 <- Dataset[which(Dataset$Class=='0'), ] 
nrow(Class0)
```
We'll take 492rows from Class1 and 492 from Class0. 
```bash
Data <- rbind(Class1, Class0[0:492,])
```
We need to shuffle.
```bash
Datashuffled <- Data[sample(nrow(Data)),]
```

# Splitting Data
We can divid our dataset into a traing and test set.
```bash
c = ncol(Datashuffled)
N = nrow(Datashuffled)
n = round(N *0.80, digits = 0)
#n = round(N *0.996, digits = 0)
trainingset = (Datashuffled[1:n, 1:c])
testset  = (Datashuffled[(n+1):N,  ])
```

Let's check the "Class" column.
```bash
counttrainingset <- table(trainingset$Class)
counttrainingset   # 393 values of 1 and 394 value of 0
```
```bash
counttestset <- table(testset$Class)
counttestset    #99 values of 1 and 98 value of 0
```
Those values can change from one person to another. It's important to add ``` set.seed(123)``` somewhere in the code to avoid having different results.

Our trainingset and testset are very well balanced.
Now, we can try to predict the Class values. We'll use 3 models: Gradient Boosted Model, SVM, and Random Forests.

--- 

# GBM: 
```bash
library(gbm)
```
```bash
gbm.model = gbm(Class~., data=trainingset, shrinkage=0.01, distribution = 'bernoulli', cv.folds=5, n.trees=2000, verbose=T)
best.iter = gbm.perf(gbm.model, method="cv")
best.iter
summary(gbm.model)
```
V3 has the highest influence among all our dataset.
```bash
GBMpred = predict(gbm.model, testset)
```
```bash
plot(GBMpred)
predictionGBM = ifelse(GBMpred>0, 'FRAUD','NOfraud')
actual = ifelse(testset$Class>0, 'FRAUD','NOfraud')
confmat <-table(predictionGBM, actual ,dnn = list('predicted', 'actual'))
print(confmat)
```
```bash
predicted FRAUD NOfraud
FRAUD      93       1
NOfraud     6      97
```
1 value was predicted as a fraud and was not. 6 values were predicted as being not a fraud and were actually one.
190 values were correctly predicted.

```bash
acc <- (confmat[2, "NOfraud"] + confmat[1, "FRAUD"]) * 100 / (confmat[2, "NOfraud"] + confmat[1, "FRAUD"] + confmat[1, "NOfraud"] + confmat[2, "FRAUD"])
# accuracy of 96.4 %
```


We can look at the ROC and AUC curves
```bash
install.packages("precrec")
library(precrec)
```

The ROC curve summerizes the false positive rate (x-axis) versus the true positive rate (y-axis).
True Positive Rate = True Positives / (True Positives + False Negatives)
False Positive Rate = False Positives / (False Positives + True Negatives)
Sensitivity = True Positives / (True Positives + False Negatives)
Specificity = True Negatives / (True Negatives + False Positives)
False Positive Rate = 1 - Specificity

```bash
ROCAUCgbm<- evalmod(scores = GBMpred, labels = testset$Class)
autoplot(ROCAUCgbm)
# 0.994 ROC 
# 0.995 Precision Recall Curve 
```
```bash
aaucs <- auc(ROCAUCgbm)
```

The PRC plots the positive predictive value (y-axis = precision) versus the true positive rate (x-axis = recall).
precision = TP/(TP+FP).
recall = TP/(TP+FN).

Here both the ROC and PRC curves are high which is a sign of a good classifier.


We can try another classifier. 
SVM have had excellent results according to the litterature. Let's give it a try.


#SVM
```bash
library(caret) # Caret has a huge list of models. Here we'll try SVM and Random forests
```

We need to have our data as factor in Caret.
```bash
trainingset2 <- trainingset
trainingset2$Class2 <- ifelse(trainingset$Class==1,'Fraud','NOfraud')
trainingset2$Class2 <- as.factor(trainingset2$Class2)
SVModel <- train(Class2 ~ ., data = trainingset2,
                 method = "svmPoly",
                 trControl= trainControl(method = "cv", number = 5) #we will be using 5-fold cross-validation.
                 
)

predSVM <- predict(SVModel, newdata = testset)
```
```bash
actual = ifelse(testset$Class>0, 'FRAUD','NOfraud')
confmatSVM <-table(predSVM, actual ,dnn = list('predicted', 'actual'))
print(confmatSVM)
```
```bash
#predicted FRAUD NOfraud
#NOfraud      0      98
#FRAUD       99       0
```
We have here 100% accuracy.

```bash
SVMp = ifelse(predSVM=="yes", 1,0)
ROCAUCSVM<- evalmod(scores = SVMp, labels = testset$Class)
autoplot(ROCAUCSVM)
```
We have a very high ROC and PRC.

```bash
aaucSVM <- auc(ROCAUCSVM)
```

# Random Forests
```bash
RFmodel <- train(Class2 ~ ., data = trainingset2,
                 method = "rf",
                 trControl= trainControl(method = "cv", number = 5), 
                 tuneGrid = expand.grid(.mtry=5) # Usually the mtry value is the SQRT of the features. Here we have 30 features so we take 5. It's the number of random variables sample at each split. 
)

predRF <- predict(RFmodel, newdata = testset)
```
```bash
actual = ifelse(testset$Class>0, 'FRAUD','NOfraud')
confmatRF <-table(predRF, actual ,dnn = list('predicted', 'actual'))
print(confmatRF)
```
```bash
#predicted FRAUD NOfraud
#nNOfraud     0      98
#FRAUD       99       0
```
We also have a 100% accuracy with Random Forests


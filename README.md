# Logistic-Regression-Project
Predict Customer Churn by Using Logistic Regression in R


## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Model Selection](#model-selection)
   
   5.1. [Goodness of fit](#goodness-of-fit)

   5.2.  [Collinearity](#collinearity)

   5.3.  [Power](#power)

   5.4. [Cross Validation](#cross-validation)

   5.5. [Odds Ratio](#odds-ratio)

   5.6. [ROC analysis](#roc-analysis)

7. [Conclusion](#conclusion)





## Introduction

Customer churn happens when customers stop using the services of a company. This problem is very important for different companies because every company needs to be aware of the customers' churn rate. It is very crucial to know the number of customers who have stopped using the company’s services because the companies should find out the reason for customer churn to make proper decisions in this matter. In this project, we are going to predict customer behavior based on the Telcom dataset of a company in California and the logistic regression model In R will be performed.


## Dataset 

The data we are using in this project is downloaded from Kaggle.com. This data included the response variable which is customer churn with different independent variables which are going to be used in this prediction. In this dataset, each row indicates a customer and each column is a feature. The target and Independent variables have been described as below:


Target variable:

Churn - Whether the customer churned or not 

Response Variables:
 
Gender - Whether the customer is a male or a female

Seniorcitizen - Whether the customer is a senior citizen or not

Partner - Whether the customer has a partner or not 

Tenure – Number of months the customer has stayed with the company

Dependents - Whether the customer has dependents or not 

Phoneservice - Whether the customer has a phone service or not

Multiplelines - Whether the customer has multiple lines or not 

Internetservice - Customer’s internet service provider 

Onlinesecurity - Whether the customer has online security or not 

Onlinebackup -Whether the customer has online backup or not 

 Deviceprotection - Whether the customer has device protection or not 
 
Techsupport - Whether the customer has tech support or not 

Streamingtv - Whether the customer has streaming TV or not 

Streamingmovies - Whether the customer has streaming movies or not 

Contract - The contract term of the customer

Paperlessbilling - Whether the customer has paperless billing or not 

Paymentmethod - The customer’s payment method 

MonthlyCharges - The amount charged to the customer monthly

TotalCharges - The total amount charged to the customer


```python

# Read the data from the "Customer Churn.csv" data file 

churn <- read.csv("Customer Churn.csv", header = TRUE)
head(churn)

# Show the structure of the data

str(churn)

```


## Data Preprocessing

- Missing Values

In the first step of data preprocessing, we removed the column that we did not need in our data analysis which is “customerID”. Then we checked the data for missing values and realized that we had missing values for the column “TotalCharges” in this case, we have chosen the complete cases of the data to have data without any missing values.  There are only 11 rows and deleting them will not affect the data. So, after removing the missing values, we now have 7032 data points in the dataset.


```python
# Finding the missing values and remove the missing values 

sapply(churn, function(x) sum(is.na(x)))
churn <- churn[complete.cases(churn), ]
churn

# Deleting the columns that we do not need in the analysis

churn$customerID <- NULL
head(churn)

churn <- churn[complete.cases(churn), ]
head(churn)
```

```python
# PRoviding the complete descriptive statistical analysis 

library(psych)
describe(churn)
```

```python
# Finding the correlation matrix 

numeric.var <- sapply(churn, is.numeric)
corr.matrix <- cor(churn[,numeric.var])
corr.matrix
```

## Exploratory Data Analysis


Based on the correlation table above, the “TotalCharges” and “MonthlyCharges” are correlated so that we can remove one of them from the data set. Here, we delete the “TotalCharges”.

```python
# Removing the variable that we dont need in the analysis after investigating the corellation

churn$TotalCharges <- NULL

```

```python

# Changing the values of the some variable to have a better undrestanding of the data 

library(plyr)

churn$Churn <- as.factor(mapvalues(churn$Churn,
                                           from=c("No","Yes"),
                                           to=c("0", "1")))
churn$OnlineSecurity <- as.factor(mapvalues(churn$OnlineSecurity,
                                            from =c("No internet service"),to=c("No")))

churn$OnlineBackup <- as.factor(mapvalues(churn$OnlineBackup,
                                            from =c("No internet service"),to=c("No")))
churn$DeviceProtection <- as.factor(mapvalues(churn$DeviceProtection,
                                            from =c("No internet service"),to=c("No")))
churn$TechSupport <- as.factor(mapvalues(churn$TechSupport,
                                            from =c("No internet service"),to=c("No")))
churn$StreamingTV <- as.factor(mapvalues(churn$StreamingTV,
                                            from =c("No internet service"),to=c("No")))
churn$StreamingMovies <- as.factor(mapvalues(churn$StreamingMovies,
                                            from =c("No internet service"),to=c("No")))
churn$MultipleLines <- as.factor(mapvalues(churn$MultipleLines,
                                             from =c("No phone service"),to=c("No")))

churn$SeniorCitizen <- as.factor(mapvalues(churn$SeniorCitizen,
                                           from=c("0","1"),
                                           to=c("No", "Yes")))

churn$Churn <- as.factor(mapvalues(churn$Churn,
                                           from=c("No","Yes"),
                                           to=c("0", "1")))

str(churn)

```



![image](https://github.com/Masoumeh89/Logistic-Regression-Project/assets/74910834/1f1d3faf-88ad-47bb-91fd-ffacd61c3a63)

Then, we noticed that some columns like “OnlineSecurity”, “OnlineBackup”, “DeviceProtection”, “TechSupport”, “streamingTV”, “streamingMovies” had three values “No”, “Yes” and “No Internet service”. So, we changed the “No Internet service” to “No”. Also, for the column “MultipleLines” we have changed the “No phone service” to “NO”. Now, we have two factors for these variables. In addition, we have changed the values in column “SeniorCitizen” from 0 or 1 to “No” or “Yes”. 

## Model Selection

In this project, a stepwise procedure has been performed for selecting the best possible models.

```python

# Model selection based on a forward stepwise method

null=glm(Churn~1,data=churn, family="binomial")
full=glm(Churn~.,data=churn, family="binomial")
step(null, scope=list(lower=null, upper=full),
       direction="forward")


```





![image](https://github.com/Masoumeh89/Logistic-Regression-Project/assets/74910834/b87f7dff-d4b6-4c45-b52a-31b7265a1aff)



Then we fitted the model with all the possible variables that have been selected by stepwise procedure.

```python
# Fitting the model to the data with logistic regression

fit.1<-glm(Churn~Contract + InternetService + tenure + PaymentMethod + 
             MultipleLines + PaperlessBilling + OnlineSecurity + StreamingMovies + 
             TechSupport + StreamingTV + SeniorCitizen + Dependents + 
             OnlineBackup,data=churn,family="binomial")
summary(fit.1)

```

![image](https://github.com/Masoumeh89/Logistic-Regression-Project/assets/74910834/d8a076cb-03d0-464d-8549-0ac25dbfc071)




### Goodness of fit

Below is the ANOVA table for the performed model:

```python

# Creating the Analysis of Deviance table

anova(fit.1, test="Chisq")

```





![image](https://github.com/Masoumeh89/Logistic-Regression-Project/assets/74910834/2cd1ce66-831f-4e3d-a2f9-70229cae13a2)

Also, we have performed the cook’s distance on the dataset and there were no influential points in the dataset because we didn’t find any large values based on the cook’s distances.  Besides, we have performed the Hosmer-Lemeshow Test to investigate the goodness of fit of the model. Based on the results, the p value is 0.7102 which means that the suggested model is a good fit.

```python

# Finding the cook's distance 

cooks.distance<-cooks.distance(fit.1)
which(cooks.distance>1)


```
```python
# Perform Hosmer-Lemeshow test 

library(ResourceSelection)
hoslem.test(fit.1$y,fitted(fit.1),g=10)

```
### Collinearity

In this section, we have checked the collinearity by using variance inflation factors. Based on this method, if any of the VIFs are greater than 10, then we can say that there is collinearity in the model. But, here we did not have any VIFs greater than 10, so there is no collinearity between the predictor variables.

```python

# Find variance inflation factors for collinearity

library(car)
vif(fit.1)




```


### Power

Another metric to investigate the accuracy of our model is the power of the model using McFadden R2. Here, the McFadden R2   is 0.2797 which is between 0.2 and 0.4, so we can conclude that the model is a good fit for predicting the customer churn. 

```python
# Find McFadden R2 value 

library(pscl)
pR2(fit.1)

```


### Cross Validation

We split the data into training and testing sections and then we fitted the Logistic Regression Model on the training data. 

Then we assessed the predictive ability of the Logistic Regression model by finding the accuracy rate. Here the Logistic Regression Accuracy is 0.80. Also, we have found the Confusion Matrix for Logistic Regression which is as below:


```python

# split the data into training and testing
 
library(caret)
Churndat<- createDataPartition(churn$Churn,p=0.7,list=FALSE)
set.seed(2022)
train<- churn[Churndat,]
test<- churn[-Churndat,]
```


```python

# Fit logistic model

fit <- glm(Churn ~ Contract + InternetService + tenure + PaymentMethod + 
             PaperlessBilling + OnlineSecurity + StreamingMovies + TechSupport + 
             StreamingTV + PhoneService + MultipleLines + SeniorCitizen + 
             Dependents + OnlineBackup,family=binomial(link="logit"),data=train)
print(summary(fit))

anova(fit, test="Chisq")

```

```python
# Find the prediction

test$Churn <- as.character(test$Churn)
pred <- predict(fit,newdata=test,type='response')
pred <- ifelse(pred > 0.5,1,0)


```

```python
# cross validation to find the accuracy 

misClasificError <- mean(pred != test$Churn)
print(paste('Logistic Regression Accuracy',1-misClasificError))


```

```python

# Confusion Matrix

print("Confusion Matrix for Logistic Regression"); 
table(test$Churn, pred > 0.5)

```


![image](https://github.com/Masoumeh89/Logistic-Regression-Project/assets/74910834/64733e59-7411-4b4e-8502-b28fe82d9b00)

### Odds Ratio

Another performance measurement in logistic regression which we have used is Odds Ratio. The odds ratio indicates the odds that an outcome will happen which has been represented below:


```python
library(MASS)
exp(cbind(OR=coef(fit), confint(fit)))

```




![image](https://github.com/Masoumeh89/Logistic-Regression-Project/assets/74910834/dd08ea83-6589-4153-a85f-20f07e3568fa)


### ROC analysis

After fitting the logistic model, Roc analysis has been performed, and the ROC curve has been plotted. As can be seen, the plot reached 1 and stays flat which means that the model is accurate. Besides, the area under the curve is 0.8401815 which is a good result because the more area under the ROC curve, the greater the accuracy. So, we can say that our model is a good fit for predicting customer churn. 

```python
# Find the ROC curve

library(ROCR)
x<-predict(fit,newdata=test,type="response")
xr<-prediction(x,test$Churn)
perform<-performance(xr,measure="tpr",x.measure="fpr")
plot(perform)
auc<-performance(xr,measure="auc")
auc<-auc@y.values[[1]]
auc

```

![image](https://github.com/Masoumeh89/Logistic-Regression-Project/assets/74910834/8a0091c9-557a-45b1-96c4-98aeb61e1b99)

## Conclusion


This project was about analyzing the customers' behavior based on the Telcom dataset of a company in California and the logistic regression model In R has been performed to predict if the customers will churn or not.  As we can see in the results, some variables like “tenure”, “Contract”, “PaperlessBilling”, “MonthlyCharges” and “InternetService” were the most important ones which affected customer churn. On the other hand, there was not any connection between gender and customer churn.

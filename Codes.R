# This is a readme file to provide the R codes of my project


# Read the data from the "Customer Churn.csv" data file 

churn <- read.csv("Customer Churn.csv", header = TRUE)
head(churn)

# Show the structure of the data

str(churn)

# Finding the missing values and remove the missing values 

sapply(churn, function(x) sum(is.na(x)))
churn <- churn[complete.cases(churn), ]
churn

# Deleting the columns that we do not need in the analysis

churn$customerID <- NULL
head(churn)

churn <- churn[complete.cases(churn), ]
head(churn)

# PRoviding the complete descriptive statistical analysis 

library(psych)
describe(churn)

# Finding the correlation matrix 

numeric.var <- sapply(churn, is.numeric)
corr.matrix <- cor(churn[,numeric.var])
corr.matrix

# Removing the variable that we dont need in the analysis after investigating the corellation

churn$TotalCharges <- NULL

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

# Model selection based on a forward stepwise method

null=glm(Churn~1,data=churn, family="binomial")
full=glm(Churn~.,data=churn, family="binomial")
step(null, scope=list(lower=null, upper=full),
     direction="forward")


# Fitting the model to the data with logistic regression

fit.1<-glm(Churn~Contract + InternetService + tenure + PaymentMethod + 
             MultipleLines + PaperlessBilling + OnlineSecurity + StreamingMovies + 
             TechSupport + StreamingTV + SeniorCitizen + Dependents + 
             OnlineBackup,data=churn,family="binomial")
summary(fit.1)


# Creating the Analysis of Deviance table

anova(fit.1, test="Chisq")

# Finding the cook's distance 

cooks.distance<-cooks.distance(fit.1)
which(cooks.distance>1)


# Perform Hosmer-Lemeshow test 

library(ResourceSelection)
hoslem.test(fit.1$y,fitted(fit.1),g=10)

# Find variance inflation factors for collinearity

library(car)
vif(fit.1)

# Find McFadden R2 value 

library(pscl)
pR2(fit.1)


# split the data into training and testing

library(caret)
Churndat<- createDataPartition(churn$Churn,p=0.7,list=FALSE)
set.seed(2022)
train<- churn[Churndat,]
test<- churn[-Churndat,]


# Fit logistic model

fit <- glm(Churn ~ Contract + InternetService + tenure + PaymentMethod + 
             PaperlessBilling + OnlineSecurity + StreamingMovies + TechSupport + 
             StreamingTV + PhoneService + MultipleLines + SeniorCitizen + 
             Dependents + OnlineBackup,family=binomial(link="logit"),data=train)
print(summary(fit))

anova(fit, test="Chisq")

# Find the prediction

test$Churn <- as.character(test$Churn)
pred <- predict(fit,newdata=test,type='response')
pred <- ifelse(pred > 0.5,1,0)

# cross validation to find the accuracy 

misClasificError <- mean(pred != test$Churn)
print(paste('Logistic Regression Accuracy',1-misClasificError))

# Confusion Matrix

print("Confusion Matrix for Logistic Regression"); 
table(test$Churn, pred > 0.5)

# Find the Odds Ratio

library(MASS)
exp(cbind(OR=coef(fit), confint(fit)))

# Find the ROC curve

library(ROCR)
x<-predict(fit,newdata=test,type="response")
xr<-prediction(x,test$Churn)
perform<-performance(xr,measure="tpr",x.measure="fpr")
plot(perform)
auc<-performance(xr,measure="auc")
auc<-auc@y.values[[1]]
auc



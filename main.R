# load the dataset
# https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression/data

#install.packages("ggplot2")
#install.packages("lmtest")
#install.packages("performanceEstimation")
#install.packages("ROSE")
#install.packages("boot")
#install.packages("caret")


library(ggplot2)
library(lmtest)
library(ROSE)
library(boot)
library(caret)
set.seed(123)


# ---------------------------------- Data Preparation -----------------------------------
# Load the data
data <- read.csv("heart.csv", header=T)

# ---------------------------------- Data Exploration -----------------------------------
# Data exploration

# Display the first few rows of the data
cat("\n ### First few rows of the data ###\n")
head(data)

# Display the number of rows and columns in the data
cat("\n ### Number of rows and columns in the data ###\n")
dim(data)

# Display the structure of the data
cat("\n ### Structure of the data ###\n")
str(data)

# Display the summary statistics of the data
cat("\n ### Summary statistics of the data ###\n")
summary(data)

#---------------------------------- Data Cleaning -----------------------------------

# Check for missing values in each column
cat("\n ### Missing values in each column ###\n")
colSums(is.na(data))

# column    missing values
# education   105
# cigsPerDay  29
# BPMeds      53
# totChol     50
# BMI         19
# heartRate   1
# glucose     388

# output the distributions of all the columns with missing values in a hystogram

# Plot the distribution of education
ggplot(data, aes(x=education)) + geom_histogram(binwidth=1, fill="skyblue", color="black") + labs(title="Distribution of Education", x="Education", y="Frequency")

# Plot the distribution of cigsPerDay
ggplot(data, aes(x=cigsPerDay)) + geom_histogram(binwidth=1, fill="skyblue", color="black") + labs(title="Distribution of Cigs Per Day", x="Cigs Per Day", y="Frequency")

# Plot the distribution of BPMeds
ggplot(data, aes(x=BPMeds)) + geom_histogram(binwidth=1, fill="skyblue", color="black") + labs(title="Distribution of BPMeds", x="BPMeds", y="Frequency")

# Plot the distribution of totChol
ggplot(data, aes(x=totChol)) + geom_histogram(binwidth=10, fill="skyblue", color="black") + labs(title="Distribution of Total Cholesterol", x="Total Cholesterol", y="Frequency")

# Plot the distribution of BMI
ggplot(data, aes(x=BMI)) + geom_histogram(binwidth=1, fill="skyblue", color="black") + labs(title="Distribution of BMI", x="BMI", y="Frequency")

# Plot the distribution of heartRate
ggplot(data, aes(x=heartRate)) + geom_histogram(binwidth=1, fill="skyblue", color="black") + labs(title="Distribution of Heart Rate", x="Heart Rate", y="Frequency")

# Plot the distribution of glucose
ggplot(data, aes(x=glucose)) + geom_histogram(binwidth=10, fill="skyblue", color="black") + labs(title="Distribution of Glucose", x="Glucose", y="Frequency")


# Impute missing values

# Impute missing values in education with the median
data$education[is.na(data$education)] <- median(data$education, na.rm=TRUE)

# Impute missing values in cigsPerDay with the median
data$cigsPerDay[is.na(data$cigsPerDay)] <- median(data$cigsPerDay, na.rm=TRUE)

# Impute missing values in BPMeds with the mode since it is a binary variable
data$BPMeds[is.na(data$BPMeds)] <- as.numeric(names(sort(table(data$BPMeds), decreasing=TRUE)[1]))

# Impute missing values in totChol with the median
data$totChol[is.na(data$totChol)] <- median(data$totChol, na.rm=TRUE)

# Impute missing values in BMI with the median
data$BMI[is.na(data$BMI)] <- median(data$BMI, na.rm=TRUE)

# Impute missing values in heartRate with the median
data$heartRate[is.na(data$heartRate)] <- median(data$heartRate, na.rm=TRUE)

# Impute missing values in glucose with the median
data$glucose[is.na(data$glucose)] <- median(data$glucose, na.rm=TRUE)

# Check for missing values after imputation
cat("\n ### Missing values in each column after imputation ###\n")
colSums(is.na(data))

#---------------------------------- Over Sampling -----------------------------------

# output the distributions of the 10YrsCHD column in a hystogram
ggplot(data, aes(x=TenYearCHD)) + geom_histogram(binwidth=1, fill="skyblue", color="black") + labs(title="Distribution of Ten Year CHD", x="Ten Year CHD", y="Frequency")

# output the number of observations with a value of 1 and 0 for TenYearCHD
cat("\n ### Number of observations with a value of 1 and 0 for TenYearCHD ###\n")
table(data$TenYearCHD)

# The data is very imbalanced, with a majority of the observations having a value of 0 for TenYearCHD
# we can fix this by oversampling the minority class using the SMOTE algorithm

# Oversample the minority class using the SMOTE algorithm
data_balanced <- ovun.sample(TenYearCHD ~ ., data=data, method="over")$data
# output the distributions of the 10YrsCHD column in a hystogram after balancing
ggplot(data_balanced, aes(x=TenYearCHD)) + geom_histogram(binwidth=1, fill="skyblue", color="black") + labs(title="Distribution of Ten Year CHD", x="Ten Year CHD", y="Frequency")

# output the number of observations with a value of 1 and 0 for TenYearCHD after balancing
cat("\n ### Number of observations with a value of 1 and 0 for TenYearCHD after balancing ###\n")
table(data_balanced$TenYearCHD)

#---------------------------------- Model Building -----------------------------------
# Model Building
# using logistic regression to predict the probability of developing coronary heart disease in the next 10 years

# Split the data into training and testing sets
# attach is used to attach the data frame to the search path
attach(data_balanced)

# create a data frame with the independent variables and the dependent variable, use all the variables
df <- data.frame(cbind(male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose, TenYearCHD))

# step1: create a logistic regression model using all the variables
logmodel <- glm(TenYearCHD ~ ., family=binomial, data=df)

# summary of the logistic regression model
cat("\n ### Summary of the logistic regression model ###\n")
summary(logmodel)

# ---------------------------------- Testing for significance -----------------------------------
# ---------------------------------- Wald test ------------------------------------------------

# qnorm(1 - 0.05/2) is used to calculate the z-value for a 95% confidence interval
cat("\n ### critical value for a 95% confidence interval ###\n")
qnorm(1 - 0.05/2)

# confidence intervals for the coefficients
cat("\n ### Confidence intervals for the coefficients ###\n")
confint(logmodel, level=0.95)

# Based on the p-values, we can see that gender, age, education, cigsPerDay, BPMeds, prevalentStroke,
# prevalentHyp, diabetes, totChol, sysBP and glucose are significant predictors of
# the probability of developing coronary heart disease in the next 10 years

# ---------------------------------- Likelihood ratio test -----------------------------------
# Step 2: Test for significance of the model using the likelihood ratio test
# Ho : The reduced model is significantly better than the full model
# H1 : The reduced model is not significantly better than the full model

# create a reduced model with only significant predictors
reduced_model <- glm(TenYearCHD ~ male + age + education + cigsPerDay + BPMeds + prevalentStroke + prevalentHyp + diabetes + totChol + sysBP + glucose, family=binomial, data=df)

# summary of the reduced model
cat("\n ### Summary of the reduced model ###\n")
summary(reduced_model)

# likelihood ratio test
cat("\n ### Likelihood ratio test ###\n")
lrtest(reduced_model, logmodel)

# calculate the critical value for a 95% confidence interval
cat("\n ### critical value for a 95% confidence interval ###\n")
qchisq(0.95, df=4)

# since the test statistic =  1.5496 < critical value = 9.487729 and
# the p-value =  0.8178 > 0.05, we fail to reject the null hypothesis
# this means that the reduced model is significantly better than the full model

# ---------------------------------- Test for Adequacy -----------------------------------
# Step 3: Calculate the R-squared value of the model, Adequacy of the model
cat("\n ### R-squared value of the model ###\n")
nullmodel <- glm(TenYearCHD ~ 1, family=binomial)
1-(reduced_model$deviance/nullmodel$deviance)

# The R-squared value of the model is 0.123 which indicates that the predictability of the model is not very good

# The AIC of the reduced model is 8698.8. Very big value when compared to the size of the dataset. This indicates that a lot of information would be lost.
# Although the reduced model is significance , the R-squared value is low and the AIC is high. This indicates that the model is not very good at predicting the probability of developing coronary heart disease in the next 10 years.

# ---------------------------------- Model Evaluation -----------------------------------
# Cross-validation
cat("\n ### Model Evaluation using Cross-validation ###\n")
# Step 4: Evaluate the model using cross-validation

# Define a function to calculate the accuracy of the model
dim(df)

control <- trainControl(method = "cv", number = 10, classProbs=TRUE)

model <- train(TenYearCHD ~ male + age + education + cigsPerDay + BPMeds + prevalentStroke + prevalentHyp + diabetes + totChol + sysBP + glucose, data=df, method="glm", trControl=control)

# output the model
cat("\n ### Model ###\n")
model

# output the resamples
cat("\n ### Resamples ###\n")
model$resample

# output the final model
cat("\n ### Final Model ###\n")
model$finalModel
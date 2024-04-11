# load the dataset
# https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression/data
# install.packages("ggplot2")
#install.packages("lmtest")
library(ggplot2)
library(lmtest)
set.seed(123)


# Load the data
data <- read.csv("heart.csv", header=T)

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

# Data Cleaning

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

# output the distributions of the 10YrsCHD column in a hystogram
ggplot(data, aes(x=TenYearCHD)) + geom_histogram(binwidth=1, fill="skyblue", color="black") + labs(title="Distribution of Ten Year CHD", x="Ten Year CHD", y="Frequency")

# # The data is very imbalanced, with a majority of the observations having a value of 0 for TenYearCHD
# # we can fix this by oversampling the minority class using the SMOTE algorithm
#
# # Oversample the minority class using the SMOTE algorithm
# data_balanced <- SMOTE(TenYearCHD ~ ., data, perc.over=100, k=5, perc.under=100)

# output the distributions of the 10YrsCHD column in a hystogram after balancing
#ggplot(data_balanced, aes(x=TenYearCHD)) + geom_histogram(binwidth=1, fill="skyblue", color="black") + labs(title="Distribution of Ten Year CHD", x="Ten Year CHD", y="Frequency")

# Model Building
# using logistic regression to predict the probability of developing coronary heart disease in the next 10 years

# Split the data into training and testing sets
# attach is used to attach the data frame to the search path
attach(data)

# create a data frame with the independent variables and the dependent variable, use all the variables
df <- data.frame(cbind(male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose, TenYearCHD))

# step1: create a logistic regression model using all the variables
logmodel <- glm(TenYearCHD ~ ., family=binomial, data=df)

# summary of the logistic regression model
cat("\n ### Summary of the logistic regression model ###\n")
summary(logmodel)

# qnorm(1 - 0.05/2) is used to calculate the z-value for a 95% confidence interval
cat("\n ### critical value for a 95% confidence interval ###\n")
qnorm(1 - 0.05/2)

# confidence intervals for the coefficients
cat("\n ### Confidence intervals for the coefficients ###\n")
confint(logmodel, level=0.95)

# Based on the p-values, we can see that gender, age, cigarettes per day, prevalentStroke, systolic blood pressure and glucose are significant predictors of the probability of developing coronary heart disease in the next 10 years

# Step 2: Test for significance of the model using the likelihood ratio test
# Ho : The reduced model is significantly better than the full model
# H1 : The reduced model is not significantly better than the full model

# create a reduced model with only significant predictors
reduced_model <- glm(TenYearCHD ~ male + age + cigsPerDay + prevalentStroke + sysBP + glucose, family=binomial, data=df)

# likelihood ratio test
cat("\n ### Likelihood ratio test ###\n")
lrtest(reduced_model, logmodel)

# calculate the critical value for a 95% confidence interval
cat("\n ### critical value for a 95% confidence interval ###\n")
qchisq(0.95, df=9-1)

# since the test statistic = 8.9956 < critical value = 15.50731 and
# the p-value = 0.4377 > 0.05, we fail to reject the null hypothesis
# this means that the reduced model is significantly better than the full model

# Step 3: Calculate the R-squared value of the model, Adequacy of the model
# R-squared = 1 - (residual deviance/reduced deviance)
cat("\n ### R-squared value of the model ###\n")
1 - (logmodel$deviance/reduced_model$deviance)
# Installing packages
packages <- c("FSA", "FSAdata", "magrittr", "dplyr", "tidyr", "plyr", "tidyverse", "psych", "Hmisc",
              "plotrix", "ggplot2", "moments", "readxl", "readr", "epiDisplay", "corrplot", "gmodels", 
              "ggpubr", "car", "AICcmodavg", "ISLR", "caret", "textreg", "pROC", "glmnet", "Metrics",
              "textreg","olsrr")

install.packages(packages)
lapply(packages, require, character.only = TRUE)

#Loading data
setwd("~/Downloads")
heartfailure<-read.csv("heart_failure_clinical_records_dataset.csv", header=TRUE,sep=",")
heartfailure_numeric<-read.csv("heart_failure_clinical_records_dataset.csv", header=TRUE,sep=",")

#------------------------------
# data library: 
# age: age of the patient (years) 
# anaemia: decrease of red blood cells or hemoglobin (boolean) 
# high blood pressure: if the patient has hypertension (boolean) 
# creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L) 
# diabetes: if the patient has diabetes (boolean) 
# ejection fraction: percentage of blood leaving the heart at each contraction (percentage) 
# platelets: platelets in the blood (kiloplatelets/mL) 
# sex: woman or man (binary) 
# serum creatinine: level of serum creatinine in the blood (mg/dL) 
# serum sodium: level of serum sodium in the blood (mEq/L) 
# smoking: if the patient smokes or not (boolean) 
# time: follow-up period (days) 
# [target] death event: if the patient deceased during the follow-up period (boolean) 

#-------------------------------

#Overview of the dataset
class(heartfailure)
sum(is.na(heartfailure))

describe(heartfailure)
summary(heartfailure)
headTail(heartfailure,6)
str(heartfailure)
dim(heartfailure)
normality(heartfailure) 
plot_normality(heartfailure)

heartfailure$DEATH_EVENT <- factor(heartfailure$DEATH_EVENT, levels = c("0","1"), labels = c("Deceased", "Alive")) 
heartfailure$DEATH_EVENT

table(heartfailure$DEATH_EVENT)

names(heartfailure)

fit <- lm(DEATH_EVENT~., heartfailure_numeric)
summary(fit)

#Creating correlation matrix
cor(heartfailure_numeric,  method = "pearson", use = "complete.obs")

pairs(DEATH_EVENT~age + ejection_fraction + serum_creatinine + time, heartfailure)

ggplot(heartfailure, aes(x = ejection_fraction, y = age, color = DEATH_EVENT)) +
  geom_jitter(width = 0, height = 0.09, alpha = 0.7)
ggplot(heartfailure,aes(x = ejection_fraction, y = serum_creatinine, color = DEATH_EVENT)) +
  geom_jitter(width = 0, height = 0.09, alpha = 0.7)
ggplot(heartfailure,aes(x = ejection_fraction, y = time, color = DEATH_EVENT)) +
  geom_jitter(width = 0, height = 0.09, alpha = 0.7)
ggplot(heartfailure,aes(x = serum_creatinine, y = age, color = DEATH_EVENT)) +
  geom_jitter(width = 0, height = 0.09, alpha = 0.7)
ggplot(heartfailure,aes(x = serum_creatinine, y = time, color = DEATH_EVENT)) +
  geom_jitter(width = 0, height = 0.09, alpha = 0.7)

#Boxplots for underlying variables: F.Undergrad, P.Undergrad, Personal, Outstate
ggplot(data=heartfailure, aes(x=ejection_fraction, y=time, fill = DEATH_EVENT)) + geom_boxplot(width=0.5)
ggplot(data=heartfailure, aes(x=ejection_fraction, y=serum_creatinine, fill = DEATH_EVENT)) + geom_boxplot(width=0.5)
ggplot(data=heartfailure, aes(x=ejection_fraction, y=age, fill = DEATH_EVENT)) + geom_boxplot(width=0.5)
ggplot(data=heartfailure, aes(x=serum_creatinine, y=age, fill = DEATH_EVENT)) + geom_boxplot(width=0.5)
ggplot(data=heartfailure, aes(x=serum_creatinine, y=time, fill = DEATH_EVENT)) + geom_boxplot(width=0.5)

view(trainsamples)
set.seed(156)
trainsamples <- heartfailure_numeric$DEATH_EVENT %>% createDataPartition (p = 0.7, list = FALSE)
train <- heartfailure_numeric[trainsamples,]
test <- heartfailure_numeric[-trainsamples, ]

x_train <- model.matrix(DEATH_EVENT~.,train)[,-1]
x_test <- model.matrix(DEATH_EVENT~., test)[,-1]

y_train <- train$DEATH_EVENT
y_test <- test$DEATH_EVENT

#Ridge regression model 
# Cross validation

set.seed(112)
cross_v_ridge <- cv.glmnet(x_train, y_train, alpha=0, nfold = 20)
cross_v_ridge

best_model_1se <- cross_v_ridge$lambda.1se
best_model_min <- cross_v_ridge$lambda.min

plot(cross_v_ridge)

# ridge regression - min
ridge_min <- glmnet(x_train, y_train, alpha = 0, lambda = best_model_min)
coef(ridge_min)
ridge_min
# #ridge regression - 1se
ridge_1se <- glmnet(x_train, y_train, alpha = 0, lambda = best_model_1se)
coef(ridge_1se)
ridge_1se

# #Performance of the fit model by RMSE ---- Train Dataset
# #RMSE performance of fit - min
RMSE_ridge_min_train_model <- predict(ridge_min, newx = x_train)
RMSE_train_ridge_min <- rmse(y_train, RMSE_ridge_min_train_model)
RMSE_train_ridge_min
# #RMSE performance of fit - 1se
RMSE_ridge_1se_train_model <- predict(ridge_1se, newx = x_train)
RMSE_train_ridge_1se <- rmse(y_train, RMSE_ridge_1se_train_model)
RMSE_train_ridge_1se

## Performance of the fit model by RMSE ---- Test Dataset
# RMSE performance of fit - min
RMSE_ridge_min_test_model <- predict(ridge_min, newx = x_test)
RMSE_test_ridge_min <- rmse(y_test, RMSE_ridge_min_test_model)
RMSE_test_ridge_min

# RMSE performance of fit - 1se
RMSE_ridge_1se_test_model <- predict(ridge_1se, newx = x_test)
RMSE_test_ridge_1se <- rmse(y_test, RMSE_ridge_1se_test_model)
RMSE_test_ridge_1se


# #LASSO regression model 
# #Cross Validation
cross_v_lasso <- cv.glmnet(x_train, y_train, alpha=1, nfolds=20)
cross_v_lasso
cross_v_lasso$lambda.min
cross_v_lasso$lambda.1se
plot(cross_v_lasso)

# #Lasso performance of fit - min
lasso_min <- glmnet(x_train, y_train, alpha = 1, lambda = cross_v_lasso$lambda.min)
lasso_min

coef(lasso_min)

# #Lasso performance of fit - 1se
lasso_1se <- glmnet(x_train, y_train, alpha = 1, lambda = cross_v_lasso$lambda.1se)
coef(lasso_1se)
lasso_1se

##Performance of fit model by RMSE ---- Train Dataset
##RMSE performance of fit - min
RMSE_lasso_min_train_model <- predict(lasso_min, newx = x_train)
RMSE_train_lasso_min <- rmse(y_train, RMSE_lasso_min_train_model)
RMSE_train_lasso_min

# #RMSE performance of fit - 1se
RMSE_lasso_1se_train_model <- predict(lasso_1se, newx = x_train)
RMSE_train_lasso_1se <- rmse(y_train, RMSE_lasso_1se_train_model)
RMSE_train_lasso_1se

# #Performance of the fit model by RMSE ---- Test Dataset
# #RMSE performance of fit - min
RMSE_lasso_min_test_model <- predict(lasso_min, newx = x_test)
RMSE_test_lasso_min <- rmse(y_test, RMSE_lasso_min_test_model)
RMSE_test_lasso_min

# #RMSE performance of fit - 1se
RMSE_lasso_1se_test_model <- predict(lasso_1se, newx = x_test)
RMSE_test_lasso_1se <- rmse(y_test, RMSE_lasso_1se_test_model)
RMSE_test_lasso_1se

#Logistic Regression Modelling


#Model 1 - Full model 
Model1 <- glm(DEATH_EVENT ~., data = train, family = binomial(link = "logit"))
screenreg(Model1)
summary(Model1)

confint(Model1)
confint.default(Model1)
exp(coef(Model1))

forward_fit_p <- ols_step_forward_p(fit, r = .05)
forward_fit_aic <- ols_step_forward_aic(fit)
forward_fit_aic
forward_fit_p

# -- Backward Selection - Least incremental predictor power
#p value - prem (p value for removal)
backward_fit_p <- ols_step_backward_p (fit, prem = .05)
backward_fit_p
#AIC 
backward_fit_aic <- ols_step_backward_aic (fit, prem = .05)
backward_fit_aic

#Stepwise regression using p-values
both_fit_p <- ols_step_both_p (fit, pent = .05, prem = .05)
both_fit_p

#Model 2 - Linear regression selection
Model2<- glm(DEATH_EVENT ~ age + serum_creatinine + ejection_fraction + time, data = train, family = binomial(link = "logit"))
screenreg(Model2)
summary(Model2)

confint(Model2)
confint.default(Model2)
exp(coef(Model2))

summary(lm(DEATH_EVENT ~ age + serum_creatinine + ejection_fraction + time, data = train))

#Model 3 - Backward
Model3 <- glm(DEATH_EVENT ~ anaemia + smoking + high_blood_pressure + platelets + diabetes, data = train, family = binomial(link = "logit"))
screenreg(Model3)
summary(Model3)

confint(Model3)
confint.default(Model3)
exp(coef(Model3))

summary(lm(DEATH_EVENT ~ anaemia + smoking + high_blood_pressure + platelets + diabetes, data = train))

#Model 4 - Forward 
Model4 <- glm(DEATH_EVENT ~ time + ejection_fraction + serum_creatinine + age + serum_sodium + creatinine_phosphokinase + sex, data = train, family = binomial(link = "logit"))
screenreg(Model4)
summary(Model4)



confint(Model4)
confint.default(Model4)

exp(coef(Model4))

summary(lm(DEATH_EVENT ~ time + ejection_fraction + serum_creatinine + age + serum_sodium + creatinine_phosphokinase + sex, data = train))

#Model 5 - Lasso 
Model5 <- glm(DEATH_EVENT ~ age + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium + time, data = train, family = binomial(link = "logit"))
screenreg(Model5)
summary(Model5)

confint(Model5)
confint.default(Model5)
exp(coef(Model5))

summary(fit)

summary(lm(DEATH_EVENT ~ age + creatinine_phosphokinase + ejection_fraction + high_blood_pressure + serum_creatinine + serum_sodium + time, data = train))

model_AIC <- AIC(Model1,Model2,Model3,Model4,Model5)
model_BIC <- BIC(Model1,Model2,Model3,Model4,Model5)

data.frame(model_AIC,model_BIC)

#Model Result:
#Model 1 : df = 13, 185.58, Model 2: df = 5, 177.95, Model 3: df = 6, AIC = 270.46, Model 4: df = 8, AIC = 178.09, Model 5: df = 8, AIC = 176.19 
#Model 2 performs the best.
install.packages("NeuralNetTools")
library(NeuralNetTools)
library(tidyr)
library(dplyr)
library(ggplot2)
library(caret)
library(pROC)

library(readxl)
Titanic_Dataset <- read_excel("Titanic-Dataset.xlsx")
summary(Titanic_Dataset)
titanic <- Titanic_Dataset

titanic$Survived <- as.factor(titanic$Survived)
titanic$Pclass <- as.factor(titanic$Pclass)
titanic$Sex<- as.factor(titanic$Sex)
titanic$SibSp<- as.factor(titanic$SibSp)
titanic$Parch <- as.factor(titanic$Parch)
titanic$Cabin <- as.factor(titanic$Cabin)
titanic$Embarked <- as.factor(titanic$Embarked)
titanic$Ticket <- as.factor(titanic$Ticket)

summary(titanic)

#checkin na values
colSums(is.na(titanic))
#replacing NA values with the mean 
titanic <- titanic %>% mutate(across(where(is.numeric), ~ replace_na(., mean(., na.rm = TRUE))))
#removed name, cabin,ticket, embarked, and passeneger ID
titanic <- (titanic [, -c(1,4,8,9,11, 12)])
##cabin and age will ot be used in this analysis

##survival base on age
ggplot(titanic, aes(x = Survived, y = Age, fill = Sex)) +
  geom_boxplot(notch = TRUE) +
  labs(
    title = "Survival based on age ",
    x = "Survived",
    y = "Age"
  ) +
  theme_minimal()

#distribution of survival
ggplot(titanic, aes(x = Survived)) +
  geom_bar(fill = "red", color = "black", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Distribution of Survivers", x = "Survived", y = "Count")


#Partition of data for holdout method
set.seed(4567)

#Create an index variable to perform a 7/30 split 
titanic_Index<- createDataPartition(titanic$Survived, p=.7, list=FALSE, times = 1)
titanic_train <- titanic[titanic_Index,]
titanic_test <- titanic[-titanic_Index,]

#Check the proportion of each origin in training and testing partitions
prop.table(table(titanic_train$Survived)) * 100
prop.table(table(titanic_test$Survived)) * 100

##SVM
trControl <- trainControl(method = 'repeatedcv',
                          number = 10,
                          repeats =  5)

svmFit <- train(Survived ~ ., data = titanic_train, 
                method = 'svmPoly',
                trControl = trControl,
                preProcess = c("center","scale"))

svmPredict <- predict(svmFit,titanic_test)

confusionMatrix(svmPredict, titanic_test$Survived, mode="everything")


#logistic regression
logitFit <- train(Survived ~ ., data = titanic_train, 
                  method = 'glm',
                  trControl = trControl,
                  family = 'binomial' )

summary(logitFit)

logitPredClass <- predict(logitFit,titanic_test)
logitPredProbs <- predict(logitFit,titanic_test,'prob')

confusionMatrix(logitPredClass, titanic_test$Survived, mode="everything")


#Neural Networks
nnetGrid <- expand.grid(size = seq(from = 1, to = 10, by = 1),
                        decay = seq(from = 0.0001, to = 0.1, by = 1))


nnetFit <- train(Survived ~ ., 
                 data = titanic_train, 
                 method = 'nnet', 
                 preProcess = c("center","scale"), 
                 trControl = trControl, 
                 tuneGrid = nnetGrid)
plot(nnetFit)

nnetPredClass <- predict(nnetFit,titanic_test)
confusionMatrix(nnetPredClass, titanic_test$Survived, mode="everything")

#NNet plot
plotnet(nnetFit, alpha = 0.6)





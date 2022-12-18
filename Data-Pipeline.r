library(dplyr)
library(tidyr)
library(caret)

df <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=FALSE)

colnames(df) <- c("sepal_length", "sepal_width", "petal_length", "petal_width", "label")
df$label <- ifelse(df$label == "Iris-setosa", 0, ifelse(df$label == "Iris-versicolor", 1, 2))

X <- select(df, -label)
y <- df$label

splitIndex <- createDataPartition(y, p=0.8, list=FALSE)
X_train <- X[splitIndex, ]
y_train <- y[splitIndex]
X_test <- X[-splitIndex, ]
y_test <- y[-splitIndex]

model <- caret::train(X_train, y_train, method = "glm", family = "multinomial")

predictions <- predict(model, X_test)

confusionMatrix(predictions, y_test)

saveRDS(model, file = "model.rds")

loaded_model <- readRDS("model.rds")

new_data <- data.frame(sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2)
prediction <- predict(loaded_model, new_data)
print(prediction)

ggplot(data = as.data.frame(X_train), mapping = aes(x = sepal_length, y = sepal_width, color = y_train)) + geom_point()

ggpairs(X_train, mapping = aes(color = y_train))

X_train <- X_train[X_train$sepal_length > 0, ]

X_train_log <- log(X_train)

X_train_log$label <- y_train

grouped_data <- group_by(X_train_log, sepal_length) %>% summarize(sepal_width = mean(sepal_width))

scaler <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(scaler, X_train)

imputer <- preProcess(X_train, method = c("knnImpute"))
X_train_imputed <- predict(imputer, X_train)

X_train_encoded <- model.matrix(~ .-1, X_train)

selector <- caret::train(X_train, y_train, method = "kknn")
X_train_selected <- predict(selector, X_train)
X_test_selected <- predict(selector, X_test)

model <- randomForest(X_train_selected, y_train)

predictions <- predict(model, X_test_selected)

accuracy <- caret::confusionMatrix(predictions, y_test)$overall[1]
print(paste("Test accuracy:", accuracy))

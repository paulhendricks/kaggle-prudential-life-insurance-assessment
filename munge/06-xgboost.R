library(readr)
library(xgboost)

# Set a random seed for reproducibility
set.seed(1)

cat("reading the train and test data\n")
train <- read_csv("../data/prepped/train.csv")
test  <- read_csv("../data/prepped/test.csv")

feature.names <- names(train)[2:ncol(train)-1]
train_names <- character(0)
cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
  if (any(is.na(train[[f]]))) {
    train[[f]] <- NULL
    test[[f]] <- NULL
  } else {
    train_names <- c(train_names, f)
  }
}

cat("training a XGBoost classifier\n")
clf <- xgboost(data = data.matrix(train[, train_names]),
               label = train$Response,
               nrounds = 100,
               objective = "reg:linear",
               eval_metric = "rmse")

cat("making predictions\n")
submission <- data.frame(Id=test$Id)
submission$Response <- as.integer(round(predict(clf, data.matrix(test[,train_names]))))

# I pretended this was a regression problem and some predictions may be outside the range
submission[submission$Response<1, "Response"] <- 1
submission[submission$Response>8, "Response"] <- 8

cat("saving the submission file\n")
write_csv(submission, "../data/prepped/submission-xgboost.csv")


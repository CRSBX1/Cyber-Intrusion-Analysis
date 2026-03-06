#==== Import Packages ===#

library(ggplot2)
library(caret)
library(randomForest)
library(xgboost)
library(PRROC)

#==== Import Dataset ====#
url = "C:\\Users\\Lenovo\\OneDrive\\Documents\\APU Second Year\\DA Programming\\Assignment\\UNSW-NB15_uncleaned.csv"
df = read.csv(url)

# Data Preprocessing
dim(df)
summary(df)
str(df)
colSums(is.na(df))


#==== Data cleaning ====#

#---- synack and ackdat ----#
View(df[is.na(df$synack),])
View(df[is.na(df$ackdat),])
sum(is.na(df$synack))
sum(is.na(df$ackdat))

sum(df[df$synack == "" & !is.na(df$synack),])
sum(df[df$ackdat == "" & !is.na(df$ackdat),])

# Symbol Contamination and NA String Check
symbol_count_synack <- grepl("[?_]",df$synack)
sum(symbol_count_synack)

NA_str_synack <- grepl("NA",df$synack)
sum(NA_str_synack)

symbol_count_ackdat <- grepl("[?_]",df$ackdat)
sum(symbol_count_ackdat)

NA_str_ackdat <- grepl("NA",df$ackdat)
sum(NA_str_ackdat)

# Symbol Substitution
df$synack <- gsub("[?_]","",df$synack)
df$ackdat <- gsub("[?_]","",df$ackdat)

# Data Conversion
df$synack <- as.numeric(df$synack)
df$ackdat <- as.numeric(df$ackdat)
class(df$synack)
class(df$ackdat)

# Check for remaining NA data
sum(is.na(df$synack))
sum(is.na(df$ackdat))

# NA values imputation

#1. Check synack and ackdat data distribution
quantile(df$synack, probs = seq(0, 1, 0.1), na.rm = TRUE)
quantile(df$ackdat, probs = seq(0, 1, 0.1), na.rm = TRUE)
ggplot(df, aes(x=synack)) + geom_boxplot()
ggplot(df, aes(x=ackdat)) + geom_boxplot()

#2. Substitute synack NA values with its 75th quantile
p75 <- quantile(df$synack, 0.75, na.rm = TRUE)
df$synack[is.na(df$synack)] <- p75
sum(is.na(df$synack))

#3. Substitute ackdat NA values with its 75th quantile
p75 <- quantile(df$ackdat, 0.75, na.rm = TRUE)
p75
df$ackdat[is.na(df$ackdat)] <- p75
sum(is.na(df$ackdat))

#---- label ----#

sum(is.na(df$label))
class(df$label)

# Check for symbols and NA strings
sum(grepl("[?_]",df$label))
sum(grepl("NA",df$label))

# Substitute symbols and NA strings
df$label <- gsub("[?_]","",df$label)
df$label <- gsub("NA",NA,df$label)


# sum of supposed normal activity labels (0) that are NA
sum( (is.na(df$label)) & (df$attack_cat=="Normal" & !is.na(df$attack_cat)) )

# sum of supposed attack activity labels (1) that are NA
sum( (is.na(df$label)) & (!df$attack_cat=="Normal" & !is.na(df$attack_cat)) )

# sum of conflicting label and attack_cat data ("Normal" attack_cat but "1" label & the opposite)
sum( (df$label=="1"  & !is.na(df$label)) & (df$attack_cat=="Normal" & !is.na(df$attack_cat)) )
sum( (df$label=="0"  & !is.na(df$label)) & (!df$attack_cat=="Normal" & !is.na(df$attack_cat)) )

# sum of NA values in both attack_cat and label
sum( is.na(df$attack_cat) & is.na(df$label) )

# substitute NA label data with their rightful values
df$label[which( (is.na(df$label)) & (df$attack_cat=="Normal" & !is.na(df$attack_cat)) )] <- "0"
df$label[which( (is.na(df$label)) & (!df$attack_cat=="Normal" & !is.na(df$attack_cat)) )] <- "1"

#==== Store Clean Data ====#
df_clean <- df[!is.na(df$label),]
sum(is.na(df_clean$label))
dim(df)
dim(df_clean)

#==== Dataset without extreme outliers ====#
df_final = subset(df_clean, synack<=0.1 & ackdat<=0.1)


#==== Data Analysis ====#

# Analysis questions:
# 1. Relationship between synack and ackdat as well as their association their association with cyber intrusion presence
# 2. synack's association with cyber intrusion presence
# 3. ackdat's association with cyber intrusion presence

#---- Q1. Relationship between synack and ackdat ----#

# scatter plot (data pointsclassified by label)
ggplot(df_final, aes(x=synack, y=ackdat, alpha = 0.1)) + geom_point(aes(color=label)) +
  geom_smooth(color = "black", fill = "grey70") +
  labs(title = "Relationship between synack and ackdat",
       x = "synack",
       y = "ackdat") + scale_color_manual(values = c("0" = "red", "1" = "lightblue"))


#---- Q2. synack's Association with cyber intrusion presence ----#

# synack boxplot
ggplot(df_final, aes(x= label, y=synack, fill = label)) + geom_boxplot() +
  labs(title = "SYN to SYN-ACK Packets Exchange Duration based on Network Traffic Type",
       x = "Classification. 0 = Normal, 1 = Cyber Intrusion",
       y = "setup time (in seconds)")


# Wilcoxon rank sum test - synack
wilcox.test(synack ~ label,data = df_final)

#---- Q3. ackdat's Association with cyber intrusion presence ----#

#ackdat boxplot
ggplot(df_final, aes(x= label, y=ackdat, fill = label)) + geom_boxplot() +
  labs(title = "SYN-ACK to ACK Packets Exchange Duration based on Network Traffic Type",
       x = "Classification. 0 = Normal, 1 = Cyber Intrusion",
       y = "setup time (in seconds)")

#Wilcoxon rank sum test - ackdat
wilcox.test(ackdat ~ label,data = df_final)


#==== Extra Feature: Data Classification using Random Forest and XGBoost ====#

#---- Data Processing ----#

# Dataset splitting
set.seed(123)
train_split = createDataPartition(df_final$label,p=0.8,list=F) #80:20 split
df_final_train = df_final[train_split,] # 80% of dataset data
df_final_test = df_final[-train_split,] # 20% of dataset data

#turn label column to factor
df_final_train$label = factor(df_final_train$label, levels = c("0","1"))
df_final_test$label = factor(df_final_test$label, levels = c("0","1"))

#turn label column from factor to numeric
train_label = as.numeric(as.character(df_final_train$label))
test_label = as.numeric(as.character(df_final_test$label))


#----Random Forest ----#

# Model Training
rfModel = randomForest(
  label ~ synack + ackdat + totalhandshake + delta,
  data = df_final_train,
  ntree = 1000,
  mtry = 2,
  nodesize = 5
)

# Classify testing data
rfPredict = predict(rfModel, newdata = df_final_test)

#Confusion matrix to measure model performance
confusionMatrix(rfPredict,df_final_test$label, positive = "1")

#---- XGBoost ----#

# Create matrices from training data
train_matrix = as.matrix(df_final_train[,c("synack","ackdat","totalhandshake","delta")])
test_matrix = as.matrix(df_final_test[,c("synack","ackdat","totalhandshake","delta")])

# Create XGB Matrices for the model's data source
dTrain = xgb.DMatrix(data = train_matrix, label = train_label)
dTest = xgb.DMatrix(data = test_matrix, label = test_label)

# Set weight and create hyperparameter grid for hyperparameter tuning
scale_weight <- sum(train_label == 0) / sum(train_label == 1)

param_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 0.1, 0.5),
  colsample_bytree = c(0.6, 0.8, 1),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.6, 0.8, 1)
)


# Hyperparameter tuning through exhaustive grid search
best_auc = 0
best_params = NULL


for(i in 1:nrow(param_grid)){
  
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i],
    gamma = param_grid$gamma[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    min_child_weight = param_grid$min_child_weight[i],
    subsample = param_grid$subsample[i],
    scale_pos_weight = scale_weight
  )
  
  cv_model = xgb.cv(
    params = params,
    data = dTrain,
    nrounds = param_grid$nrounds[i],
    nfold = 5,
    verbose = 0,
    early_stopping_rounds = 10
  )
  
  max_auc = max(cv_model$evaluation_log$test_auc_mean)
  
  if(max_auc > best_auc){
    best_auc = max_auc
    best_params = list(
      params = params,
      nrounds = cv_model$best_iteration
    )
  }
  cat(sprintf("Iteration %d/%d - AUC: %.4f\n", i, nrow(param_grid), max_auc))
}

# Train final XGBoost model
xgb_model = xgboost(
  data = dTrain,
  params = best_params$params,
  nrounds = best_params$nrounds,
  verbose = 1
)

# Classify testing data
xgb_pred = predict(xgb_model,dTest)
xgb_pred_class = factor(ifelse(xgb_pred > 0.5,1,0))

# Confusion matrix to measure model performance
confusionMatrix(xgb_pred_class, factor(df_final_test$label), positive = "1")


#---- Precision-Recall (PR) Curve ----#
# PR Curve needs prediction probability to measure the model's precision and recall
# XGBoost's classification/prediction results returns probabilities, meanwhile RF returns factors on default

# Set RF's prediction type to "prob" so RF returns probabilities
pred_prob = predict(rfModel, df_final_test, type = "prob")[,2]

# Create PR curve objects of each model's classfication result
pr_obj = pr.curve(scores.class0 = pred_prob,
                  weights.class0 = test_label == 0,
                  curve = T)

pr_xgb = pr.curve(scores.class0 = xgb_pred,
                  weights.class0 =  test_label == 0,
                  curve = T)

# Create a combined dataset that contains PR curve data features
pr_combined = rbind(
  data.frame(
    recall = pr_obj$curve[,1],
    precision = pr_obj$curve[,2],
    threshold = pr_obj$curve[,3],
    model = "Random Forest"
  ),
  data.frame(
    recall = pr_xgb$curve[,1],
    precision = pr_xgb$curve[,2],
    threshold = pr_xgb$curve[,3],
    model = "xGBoost"
  ))

# Create a F1 score column to verify each threshold's peformance
pr_combined$f1 = 2*pr_combined$precision*pr_combined$recall/(pr_combined$recall+pr_combined$precision)
pr_optimal_rf = pr_combined %>%
  filter(model == "Random Forest") %>%
  slice_max(f1, n=1, with_ties = F)
pr_optimal_xgb = pr_combined %>%
  filter(model=="xGBoost") %>%
  slice_max(f1, n = 1, with_ties = F)


# Create a PR curve plot with the combined dataset as the data source
ggplot(pr_combined, aes(x = recall, y = precision, color = model)) +
  geom_line(linewidth = 1) +
  # Draw RF's optimal point
  annotate("point", x = pr_optimal_rf$recall, y = pr_optimal_rf$precision, size = 3, color = "red") +
  annotate("text", x = pr_optimal_rf$recall - 0.05, y = pr_optimal_rf$precision - 0.015, fontface = "bold", color = "red",
           label = paste0("RF's Optimal\n", round(pr_optimal_rf$threshold,3)),
           vjust = 0.5, hjust = 0.5) +
  # Draw xGBoost's optimal point
  annotate("point", x = pr_optimal_xgb$recall, y = pr_optimal_xgb$precision, size = 3, color = "blue") +
  annotate("text", x = pr_optimal_xgb$recall - 0.07, y = pr_optimal_xgb$precision - 0.045, color = "blue",
           fontface = "bold", label = paste0("xGB's Optimal \n", round(pr_optimal_xgb$threshold,3)),
           vjust = 0.5, hjust = 0.5) +
  labs(title = "Random Forest vs xGBoost Precision-Recall (PR) Curve",
       subtitle = paste0("RF F1 Score = ", round(pr_optimal_rf$f1,3),", XGBoost F1 Score = ",
                         round(pr_optimal_xgb$f1,3)),
       x = "Recall",
       y = "Precision")

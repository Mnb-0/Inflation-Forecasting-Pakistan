# Load necessary libraries
library(tidyverse)
library(readr)
library(reshape2)
library(GGally)
library(forecast)
library(glmnet)
library(tidyr)
library(zoo)

# Load dataset
data <- read_csv("dataset.csv")
print(data)

# Transpose and clean
data_t <- data %>%
  select(-`Series Code`) %>%
  column_to_rownames("Series Name") %>%
  t() %>%
  as.data.frame()
# Convert year row to column
data_t$Year <- rownames(data_t)
data_t$Year <- as.integer(str_extract(data_t$Year, "\\d{4}"))
colnames()
# Replace '..' with NA
 data_t_clean <- data_t %>%
   mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Convert character columns to numeric
data_t <- data_t %>%
  mutate(across(where(is.character), ~ as.numeric(.)))

# Clean column names
names(data_t) <- make.names(names(data_t))

# Impute missing values with mean
data_t_clean <- data_t %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))


print(data$`GDP growth (annual %)`)

# --- Summary Statistics ---
summary_stats <- data_t_clean %>%
  summarise(across(where(is.numeric), 
                   list(Mean = ~ mean(. , na.rm = TRUE),
                        Median = ~ median(. , na.rm = TRUE),
                        Mode = ~ as.numeric(names(sort(table(.), decreasing = TRUE))[1]),
                        Q1 = ~ quantile(., 0.25, na.rm = TRUE),
                        Q3 = ~ quantile(., 0.75, na.rm = TRUE),
                        IQR = ~ IQR(. , na.rm = TRUE)
                   ), .names = "{col}_{fn}"))

# View summary statistics
print(summary_stats)

# --- PNG Output for Plots ---
png("Boxplot_All_Variables.png", width = 1200, height = 800)
data_long <- data_t_clean %>%
  pivot_longer(cols = -Year, names_to = "Variable", values_to = "Value")

# Boxplot for all numeric variables
ggplot(data_long, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "skyblue", outlier.color = "red", outlier.shape = 8) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplot of All Variables", x = "Variable", y = "Value")
dev.off()

# --- Selecting the top 10 significant variables ---
top10_vars <- c(
  "GDP.growth..annual...", 
  "Official.exchange.rate..LCU.per.US...period.average.",
  "Imports.of.goods.and.services....of.GDP.", 
  "Trade....of.GDP.",
  "Industry..including.construction...value.added....of.GDP.",
  "GDP.per.capita.growth..annual...",
  "Claims.on.central.government..annual.growth.as...of.broad.money.",
  "Gross.capital.formation..annual...growth.",
  "Industry..including.construction...value.added..annual...growth.",
  "Manufacturing..value.added..annual...growth."
)

# Create modeling dataset
df_model <- data_t_clean %>%
  select(Year, Inflation = Inflation..consumer.prices..annual..., all_of(top10_vars)) %>%
  arrange(Year) %>%
  mutate(
    Inflation_Lag1 = lag(Inflation, 1),
    Inflation_Lag2 = lag(Inflation, 2),
    Inflation_RollMean3 = rollmean(Inflation, k = 3, fill = NA, align = "right")
  ) %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Pairplot of top correlated variables
df_pairplot <- df_model %>%
  select(Inflation, all_of(top10_vars))

png("Pairplot_Top_Correlated_Variables.png", width = 1200, height = 800)
print(ggpairs(df_pairplot, title = "Pairplot of Top 10 Correlated Variables with Inflation"))
dev.off()

# --- Feature/target setup ---
x <- as.matrix(df_model %>% select(-Year, -Inflation))
y <- df_model$Inflation

# ---- Train/Test Split ----
set.seed(42)
n <- nrow(df_model)
train_size <- round(0.8 * n)
train_index <- 1:train_size
test_index <- (train_size + 1):n

x_train <- x[train_index, ]
y_train <- y[train_index]
x_test <- x[test_index, ]
y_test <- y[test_index]

# ---- Train Models ----

# ARIMA
fit_arima <- arima(y_train, order = c(2, 0, 2))
arima_forecast <- forecast(fit_arima, h = length(y_test))
arima_pred <- as.numeric(arima_forecast$mean)

# Ridge
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)
ridge_pred <- predict(cv_ridge, s = "lambda.min", newx = x_test)

# LASSO
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
lasso_pred <- predict(cv_lasso, s = "lambda.min", newx = x_test)

# Elastic Net
cv_elastic <- cv.glmnet(x_train, y_train, alpha = 0.5)
elastic_pred <- predict(cv_elastic, s = "lambda.min", newx = x_test)

# ---- OLS Regression for P-values ----
df_train <- as.data.frame(x_train)
df_train$Inflation <- y_train

# Fit standard linear model
lm_model <- lm(Inflation ~ ., data = df_train)

# Summary includes p-values
cat("\n--- OLS Model Summary (for p-values) ---\n")
summary_lm <- summary(lm_model)
print(summary_lm)

# Extract p-values
p_values <- summary_lm$coefficients[, "Pr(>|t|)"]
cat("\nP-values of predictors:\n")
print(p_values)

# ---- Model Summaries ----
cat("\n--- ARIMA Model Summary ---\n")
print(summary(fit_arima))

cat("\n--- Ridge Regression Summary ---\n")
cat("Optimal Lambda (Ridge):", cv_ridge$lambda.min, "\n")
ridge_coef <- coef(cv_ridge, s = "lambda.min")
print(ridge_coef)

cat("\n--- LASSO Regression Summary ---\n")
cat("Optimal Lambda (LASSO):", cv_lasso$lambda.min, "\n")
lasso_coef <- coef(cv_lasso, s = "lambda.min")
print(lasso_coef)

cat("\n--- Elastic Net Regression Summary ---\n")
cat("Optimal Lambda (Elastic Net):", cv_elastic$lambda.min, "\n")
elastic_coef <- coef(cv_elastic, s = "lambda.min")
print(elastic_coef)

# ---- Evaluation ----
mse <- function(actual, predicted) mean((actual - predicted)^2)
r2_score <- function(actual, predicted) {
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  1 - ss_res / ss_tot
}

mse_test <- data.frame(
  Model = c("ARIMA", "Ridge", "LASSO", "ElasticNet"),
  MSE = c(
    mse(y_test, arima_pred),
    mse(y_test, ridge_pred),
    mse(y_test, lasso_pred),
    mse(y_test, elastic_pred)
  )
)

r2_test <- data.frame(
  Model = c("ARIMA", "Ridge", "LASSO", "ElasticNet"),
  R2 = c(
    r2_score(y_test, arima_pred),
    r2_score(y_test, ridge_pred),
    r2_score(y_test, lasso_pred),
    r2_score(y_test, elastic_pred)
  )
)

# Print metrics to console
print("Test Set Mean Squared Errors:")
print(mse_test)

print("Test Set R-squared Values:")
print(r2_test)

# --- Boxplot for all numeric variables ---
data_long <- data_t_clean %>%
  pivot_longer(cols = -Year, names_to = "Variable", values_to = "Value")

# Boxplot for all numeric variables
ggplot(data_long, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "skyblue", outlier.color = "red", outlier.shape = 8) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplot of All Variables", x = "Variable", y = "Value")


# plotsss
# Variables with larger values
large_value_vars <- c(
  "Foreign.direct.investment..net.inflows....of.GDP.",
  "GDP.per.capita.growth..annual..."
)

# Variables with smaller values
small_value_vars <- setdiff(names(data_t_clean), large_value_vars)

# Create a boxplot for the larger value variables
data_long_large <- data_t_clean %>%
  pivot_longer(cols = large_value_vars, names_to = "Variable", values_to = "Value")

ggplot(data_long_large, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "lightgreen", outlier.color = "red", outlier.shape = 8) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplot of Variables with Larger Values", x = "Variable", y = "Value")

# Create a boxplot for the smaller value variables
data_long_small <- data_t_clean %>%
  pivot_longer(cols = small_value_vars, names_to = "Variable", values_to = "Value")

ggplot(data_long_small, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red", outlier.shape = 8) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplot of Variables with Smaller Values", x = "Variable", y = "Value")

# --- Pairplot of top correlated variables ---
df_pairplot <- df_model %>%
  select(Inflation, all_of(top10_vars))

# Pairplot of top correlated variables
ggpairs(df_pairplot, title = "Pairplot of Top 10 Correlated Variables with Inflation")

# --- Prediction vs Actual Inflation Plot ---
df_plot <- data.frame(
  Year = df_model$Year[test_index],
  Actual = y_test,
  ARIMA = arima_pred,
  Ridge = as.numeric(ridge_pred),
  LASSO = as.numeric(lasso_pred),
  ElasticNet = as.numeric(elastic_pred)
)

df_long <- pivot_longer(df_plot, cols = -Year, names_to = "Model", values_to = "Inflation")

# Prediction plot
ggplot(df_long, aes(x = Year, y = Inflation, color = Model)) +
  geom_line(linewidth = 1.2) +  # Changed 'size' to 'linewidth'
  theme_minimal() +
  labs(title = "Model Predictions vs Actual Inflation (Test Set)", y = "Inflation (%)")

# --- Correlation Matrix Calculation ---
cor_matrix <- df_pairplot %>%
  cor(use = "complete.obs") 

# Print the correlation matrix
print("Correlation Matrix:")
print(round(cor_matrix, 2))

# --- Heatmap of Correlation Matrix ---
# Convert to long format for ggplot
cor_df <- as.data.frame(cor_matrix) %>%
  rownames_to_column(var = "Var1") %>%
  pivot_longer(-Var1, names_to = "Var2", values_to = "Correlation")

# Plot the heatmap
ggplot(cor_df, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0,
                       limit = c(-1, 1), name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(angle = 0)) +
  labs(title = "Correlation Matrix Heatmap", x = "", y = "") +
  coord_fixed()

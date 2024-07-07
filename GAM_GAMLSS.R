
# Train a model on UKB data. Test on other datasets. 
library(dplyr)
library(mgcv)
library(gamlss)
library(abind)
library(pROC)
library(mgcv)
library(Metrics)
library(ggplot2)

model_gam <-  "GAM"
model_gamlss <- "GAMLSS" # GAMLSS returns negative sigma
merged <- read.csv(file="path/to/combined_data_file.csv")
#lh_columns <- names(data)[grepl("^(lh|rh)_.*thickness$", names(data))]
data_patients <- read.csv(file="all_AD_MCI_subjects.csv")
data_train <- read.csv(file="path/to/CN_train_split.csv")
data_test <- read.csv(file="path/to/CN_test_split.csv")

AD_ROI = c('lh_entorhinal_thickness','lh_inferiortemporal_thickness','lh_middletemporal_thickness','lh_inferiorparietal_thickness','lh_fusiform_thickness') #,'lh_precuneus_thickness')


resultsDir <- '~/Desktop/SS2024/IDP/resultTables/test'

# data preparation
data <- merged
data <- data[data$DX %in% c("CN","AD","MCI"),]
data$DX <- factor(data$DX)
data <- data[,c(1:2,7:106)]

# GAM and GAMLSS: Age & Sex models
mydata <- data # problems of GAMLSS with variable "data"

# estimate models GAM and GAMLSS from training data

gam_models <- setNames(lapply(AD_ROI, function(col) {
  gam(as.formula(paste(col, "~ s(AGE) + PTGENDER")), data = data_train, method = "REML")
}),AD_ROI)

gamlss_models <- setNames(lapply(AD_ROI, function(col) {
    gamlss(formula = as.formula(paste(col, "~ bfp(AGE, -2) + bfp(AGE, -2) * log(AGE) + PTGENDER")),
           sigma.formula = ~ bfp(AGE, -1) + bfp(AGE, 0.5) + PTGENDER,
           nu.formula = ~ 1,
           data = data_train)
}), AD_ROI)



# Calculate residuals and Z-scores
calculate_residuals_and_zscores <- function(models, data_test, data, AD_ROI, model_type) {
  errors <- list()
  errors_healthy <- list()
  for (col in names(models)) {
    errors[[col]] <- predict(models[[col]], newdata = data)
    errors_healthy[[col]] <- predict(models[[col]], newdata = data_test)
  }
  average_predictions <- do.call(cbind, errors)
  average_prediction_healthy <- do.call(cbind, errors_healthy)
  residual <- data_test[, AD_ROI] - average_prediction_healthy
  residual_all <- data[, AD_ROI] - average_predictions

  if (model_type == "GAMLSS") {
    gamlss_var <- list()
    for (col in names(models)) {
      gamlss_var[[col]] <- predict(models[[col]], what = 'sigma', newdata = data)
      print(mean(gamlss_var[[col]]))
    }
    var <- do.call(cbind, gamlss_var)
    z_scores <- residual_all / abs(var)
  } else {
    var <- apply(residual, 2, sd, na.rm = TRUE)
    print((var))
    z_scores <- sweep(residual_all, 2, var, FUN = '/')
  }
  z_scores <- cbind(data[, 3:4], z_scores)
  residual_all <- cbind(data[, 3:4], residual_all)
  return(list(z_scores = z_scores, residual_all = residual_all, var = var, avg_pred=average_predictions))
}


# Define the compute_statistics function
compute_statistics <- function(models, data_te, suffix = "", experiment="",resultsDir = "") {
  
  # Helper function to compute explained variance
  compute_explained_variance <- function(y_true, y_pred) {
    variance_of_residuals <- var(y_true - y_pred)
    variance_of_true <- var(y_true)
    explained_variance <- 1 - (variance_of_residuals / variance_of_true)
    return(explained_variance)
  }
  
  # Calculate the mean squared error (MSE)
  errors_healthy <- list()
  errors_healthy_mean <- list()
  logLossGaussian <- function(actual, predicted, sd) {
    -mean(dnorm(actual, mean = predicted, sd = sd, log = TRUE))
  }
  
  # Initialize list to store MSLL values
  msll_list <- list()
  
  # Loop through each model
  for (col in names(models)) {
    model <- models[[col]]
    
    # Check if the model is valid (assuming valid models have a class attribute)
    if (!is.null(attr(model, "class"))) {
      # Generate predictions
      predictions <- predict(model, newdata = data_te, type = "response")
      
      # Check if predictions are returned
      if (is.null(predictions) || length(predictions) != nrow(data_te)) {
        warning(paste("No valid predictions for model", col))
        next
      }
      # Get actual values
      actual_values <- data_te[[col]]
      
      # Calculate prediction errors
      errors_healthy[[col]] <- predictions - actual_values
      # Calculate mean absolute error
      errors_healthy_mean[[col]] <- mean(abs(errors_healthy[[col]]))
      
      # Calculate residuals
      residuals <- actual_values - predictions
      
      # Calculate the standard deviation of the residuals
      sd_residuals <- sd(residuals)
      
      # Calculate log loss for the model
      model_log_loss <- logLossGaussian(actual_values, predictions, sd_residuals)

      # Calculate mean of actual values (null model prediction)
      mean_actual <- mean(actual_values)
      # Calculate log loss for the null model
      null_log_loss <- logLossGaussian(actual_values, rep(mean_actual, length(actual_values)), sd_residuals)
      # Calculate MSLL
      msll <- model_log_loss - null_log_loss
      msll_list[[col]] <- msll
    } else {
      warning(paste("Invalid model for column", col))
    }
  }
  # Convert the list of mean errors to a vector for easier display
  mean_errors <- unlist(errors_healthy_mean)
  # Convert the list of MSLL values to a vector for easier display
  msll <- unlist(msll_list)
  # Compute explained variance for all models
  explained_variance <- sapply(names(models), function(col) {
    y_true <- data_te[[col]]
    y_pred <- predict(models[[col]], newdata = data_te, type = "response")
    compute_explained_variance(y_true, y_pred)
  })
  
  # Initialize a list to store BIC values for each model
  bic_values <- list()
  for (col in names(models)) {
    # Get the fitted model for the current column
    model <- models[[col]]
    
    # Make predictions using the current model
    predictions <- predict(model, newdata = data_te)
    
    # Calculate residuals for the current model
    residuals <- data_te[[col]] - predictions
    
    # Compute the log-likelihood of the new data
    logLik_new <- sum(dnorm(data_te[[col]], mean = predictions, sd = sd(residuals), log = TRUE))
    
    # Number of parameters in the model
    k <- length(coef(model))
    
    # Number of observations in the new data
    n <- nrow(data_te)
    
    # Compute BIC
    bic_new <- k * log(n) - 2 * logLik_new
    
    # Store the BIC value in the list
    bic_values[[col]] <- bic_new
  }
  # Convert the list of BIC values to a vector for easier display
  bic_vector <- unlist(bic_values)
  
  # Prepare result statistics data frame
  res_stats <- data.frame(mse = mean_errors, bic = bic_vector, msll = msll, ev = explained_variance)
  print(res_stats)
  # Save results to a CSV file
  results_file <- paste0(resultsDir, 'stats_', experiment, suffix, '.csv')
  write.csv(x = res_stats, file = results_file)
}


summarize_and_save_results <- function(z_scores, residual_all, AD_ROI, numerical_cols, resultsDir, suffix) {
  results <- residual_all %>%
    group_by(set, DX) %>%
    summarize(across(all_of(numerical_cols), list(mean = ~mean(., na.rm = TRUE), std = ~sd(., na.rm = TRUE)), .names = "{.col}_{.fn}")) %>%
    ungroup()
  write.csv(x = results, file = paste0(resultsDir, 'Mean+std_healthytrain_', suffix, '.csv'))
  
  results_mae <- residual_all %>%
    group_by(set, DX) %>%
    summarize(across(all_of(numerical_cols), ~mean(abs(.), na.rm = TRUE), .names = "{.col}_{.fn}")) %>%
    ungroup()
  
  matched_column_names <- unlist(lapply(AD_ROI, function(pattern) {grep(pattern, names(results_mae), value = TRUE)}))
  results_mae$ROI <- rowMeans(results_mae[, matched_column_names])
  results_mae <- results_mae %>% relocate(ROI, .after = DX)
  write.csv(x = results_mae, file = paste0(resultsDir, 'MAE_', suffix, '.csv'))
  
  residual_all$err <- rowMeans(abs(residual_all[, numerical_cols]))
  rec_error_grouped <- residual_all %>%
    group_by(set, DX) %>%
    dplyr::summarize(
      rec_error_mean = mean(err, na.rm = TRUE),
      rec_error_std = sd(err, na.rm = TRUE),
      .groups = "drop"
    )
  write.csv(x = rec_error_grouped, file = paste0(resultsDir, 'Total_MAE_', suffix, '.csv'))
  
  results_count <- z_scores %>%
    group_by(set, DX) %>%
    summarize(across(all_of(numerical_cols),
                     ~sum(. < -2, na.rm = TRUE) / n() * 100,
                     .names = "{.col}_count")) %>%
    ungroup()
  matched_column_names <- unlist(lapply(AD_ROI, function(pattern) {grep(pattern, names(results_count), value = TRUE)}))
  results_count$ROI <- rowMeans(results_count[, matched_column_names])
  results_count <- results_count %>% relocate(ROI, .after = DX)
  write.csv(x = results_count, file = paste0(resultsDir, 'Perc_', suffix, '.csv'))
  return(matched_column_names)
}

# Function to calculate R-squared for a subset
calculate_r_squared <- function(subset_data, average_predictions, lh_columns) {
  sst <- colSums((subset_data[, lh_columns] - colMeans(subset_data[, lh_columns]))^2)
  sse <- colSums((subset_data[, lh_columns] - average_predictions)^2)
  r_squared <- 1 - (sse / sst)
  return(r_squared)
}

# Function to compute correlation and R-squared
compute_correlation_and_r_squared <- function(z_scores, AD_ROI, sd_pop, data, average_predictions, resultsDir, suffix, matched_column_names) {
  res_stats <- data.frame(spearman = NA, kendall = NA, Rsquared = NA)
  selected_rows <- z_scores[z_scores$set %in% c('adni', 'aibl', 'jadni', 'delcode'), ]
  matched_column_names_no_suffix <- sub("_count$", "", matched_column_names)
  matched_column_names_selected <- matched_column_names_no_suffix[matched_column_names_no_suffix %in% names(selected_rows)]
  selected_rows$ROI <- rowMeans(selected_rows[, matched_column_names_selected], na.rm = TRUE)
  res_stats$spearman <- cor(as.integer(factor(selected_rows$DX, levels = c("CN", "MCI", "AD"), ordered = TRUE)), selected_rows$ROI, method = "spearman")
  res_stats$kendall <- cor(as.integer(factor(selected_rows$DX, levels = c("CN", "MCI", "AD"), ordered = TRUE)), selected_rows$ROI, method = "kendall")
  
  r_squared_results <- list()
  for (set_name in unique(data$set)) {
    r_squared_results[[set_name]] <- calculate_r_squared(data[data$set == set_name, ], average_predictions[data$set == set_name, ], AD_ROI)
  }
  r_squared_results <- do.call(cbind, r_squared_results)
  res_stats$Rsquared <- mean(r_squared_results[, c('adni', 'aibl', 'jadni', 'delcode')])
  write.csv(x = res_stats, file = paste0(resultsDir, 'STATS_', suffix, '.csv'))
}

# Function to generate plots
generate_plots_gam <- function(models, average_predictions, AD_ROI, data_healthy, resultsDir, suffix, data_patients, sd_pop) {

  remove_outliers <- function(values, threshold = 7) {
    z_scores <- (values - mean(values, na.rm = TRUE)) / sd(values, na.rm = TRUE)
    return(abs(z_scores) < threshold & values > 0)
  }
  
  plot_data_list <- list()
  for (roi in names(models)) {
    model <- models[[roi]]
    pred <- predict(model, newdata = data_healthy, se.fit = TRUE)
    predicted_values <- pred$fit
    predicted_sds <- pred$se.fit
    true_values <- data_patients[[roi]]
    non_outliers <- remove_outliers(true_values, outlier_thresh)
    filtered_data <- data_patients[non_outliers, ]
    filtered_predicted_values <- predicted_values[non_outliers]
    filtered_predicted_sds <- predicted_sds[non_outliers]
    plot_data <- data.frame(
      AGE = data_healthy$AGE,
      Prediction = predicted_values,
      SD = predicted_sds,
      ROI = roi
    )
    true_data <- data.frame(
      AGE = filtered_data$AGE,
      TrueValue = filtered_data[[roi]],
      ROI = roi
    )
    combined_data <- merge(plot_data, true_data, by = c("AGE", "ROI"))
    plot_data_list[[roi]] <- combined_data
  }
  combined_plot_data <- do.call(rbind, plot_data_list)
  if (suffix == "GAM") {
    combined_plot_data$ROI_sd_pop <- sapply(combined_plot_data$ROI, function(x) sd_pop[x])
  } else {
    combined_plot_data$ROI_sd_pop <- sapply(combined_plot_data$ROI, function(x) pred$se.fit)
  }
  
  combined_plot_data$Lower_25 <- combined_plot_data$Prediction - 1.150 * combined_plot_data$ROI_sd_pop
  combined_plot_data$Upper_25 <- combined_plot_data$Prediction + 1.150 * combined_plot_data$ROI_sd_pop
  combined_plot_data$Lower_05 <- combined_plot_data$Prediction - 1.645 * combined_plot_data$ROI_sd_pop
  combined_plot_data$Upper_05 <- combined_plot_data$Prediction + 1.645 * combined_plot_data$ROI_sd_pop
  
  ggplot(combined_plot_data, aes(x = AGE)) +
    geom_line(aes(y = Prediction, color = ROI)) +
    geom_ribbon(aes(ymin = Lower_05, ymax = Upper_05, fill = ROI), alpha = 0.1, color = NA) +
    geom_ribbon(aes(ymin = Lower_25, ymax = Upper_25, fill = ROI), alpha = 0.1, color = NA) +
    geom_point(aes(y = TrueValue, color = ROI), size = 1.5, shape = 21) +
    labs(title = "Predictions with Age-Dependent Standard Deviations and Confidence Intervals",
         x = "Age",
         y = "Thickness") +
    facet_wrap(~ ROI, scales = "free_y") +
    theme_minimal()
  
  ggplot(combined_plot_data, aes(x = AGE)) +
    geom_line(aes(y = Prediction, color = ROI)) +
    geom_ribbon(aes(ymin = Lower_05, ymax = Upper_05, fill = ROI), alpha = 0.1, color = NA) +
    geom_ribbon(aes(ymin = Lower_25, ymax = Upper_25, fill = ROI), alpha = 0.1, color = NA) +
    labs(title = "Predictions with Age-Dependent Standard Deviations and Confidence Intervals",
         x = "Age",
         y = "Thickness") +
    facet_wrap(~ ROI, scales = "free_y") +
    theme_minimal()
}
# Function to generate plots
generate_plots_gamlss <- function(models, average_predictions, AD_ROI, data_healthy, resultsDir, suffix, data_patients) {
  
  remove_outliers <- function(values, threshold = 7) {
    z_scores <- (values - mean(values, na.rm = TRUE)) / sd(values, na.rm = TRUE)
    return(abs(z_scores) < threshold & values > 0)
  }
  
  plot_data_list <- list()
  for (roi in names(models)) {
    model <- models[[roi]]
    
    # Predict using GAMLSS
    pred <- predict(model, newdata = data_healthy, what = "mu", type = "response")
    predicted_sigma <- predict(model, newdata = data_healthy, what = "sigma", type = "response")
    
    predicted_values <- pred
    predicted_sds <- predicted_sigma

    true_values <- data_patients[[roi]]
    non_outliers <- remove_outliers(true_values, outlier_thresh)
    filtered_data <- data_patients[non_outliers, ]
    filtered_predicted_values <- predicted_values[non_outliers]
    filtered_predicted_sds <- predicted_sds[non_outliers]
    
    plot_data <- data.frame(
      AGE = data_healthy$AGE,
      Prediction = predicted_values,
      SD = predicted_sds,
      ROI = roi
    )
    
    true_data <- data.frame(
      AGE = filtered_data$AGE,
      TrueValue = filtered_data[[roi]],
      ROI = roi
    )
    
    combined_data <- merge(plot_data, true_data, by = c("AGE", "ROI"))
    plot_data_list[[roi]] <- combined_data
  }
  
  combined_plot_data <- do.call(rbind, plot_data_list)
  
  # Using the predicted sigma values directly
  combined_plot_data$ROI_sd_pop <- combined_plot_data$SD
  
  combined_plot_data$Lower_25 <- combined_plot_data$Prediction - 1.150 * combined_plot_data$ROI_sd_pop
  combined_plot_data$Upper_25 <- combined_plot_data$Prediction + 1.150 * combined_plot_data$ROI_sd_pop
  combined_plot_data$Lower_05 <- combined_plot_data$Prediction - 1.645 * combined_plot_data$ROI_sd_pop
  combined_plot_data$Upper_05 <- combined_plot_data$Prediction + 1.645 * combined_plot_data$ROI_sd_pop
  
  ggplot(combined_plot_data, aes(x = AGE)) +
    geom_line(aes(y = Prediction, color = ROI)) +
    geom_ribbon(aes(ymin = Lower_05, ymax = Upper_05, fill = ROI), alpha = 0.1, color = NA) +
    geom_ribbon(aes(ymin = Lower_25, ymax = Upper_25, fill = ROI), alpha = 0.1, color = NA) +
    geom_point(aes(y = TrueValue, color = ROI), size = 1.5, shape = 21) +
    labs(title = "Predictions with Model Variance and Confidence Intervals",
         x = "Age",
         y = "Thickness") +
    facet_wrap(~ ROI, scales = "free_y") +
    theme_minimal()
  
  ggplot(combined_plot_data, aes(x = AGE)) +
    geom_line(aes(y = Prediction, color = ROI)) +
    geom_ribbon(aes(ymin = Lower_05, ymax = Upper_05, fill = ROI), alpha = 0.1, color = NA) +
    geom_ribbon(aes(ymin = Lower_25, ymax = Upper_25, fill = ROI), alpha = 0.1, color = NA) +
    labs(title = "Predictions with Model Variance and Confidence Intervals",
         x = "Age",
         y = "Thickness") +
    facet_wrap(~ ROI, scales = "free_y") +
    theme_minimal()
}


# GAM 
residuals_and_zscores <- calculate_residuals_and_zscores(gam_models, data_test, mydata, AD_ROI, "GAM")
# compute on healthy test set
compute_statistics(models = gam_models, data_te = data_test,suffix = "GAM", experiment = "healthy", resultsDir = resultsDir)
# compute on all data: 
compute_statistics(models = gam_models, data_te = mydata, suffix = "GAM", experiment = "all", resultsDir = resultsDir)
z_scores <- residuals_and_zscores$z_scores
residual_all <- residuals_and_zscores$residual_all
numerical_cols <- names(z_scores)[-(1:2)]
sd_pop <- residuals_and_zscores$var
average_predictions <- residuals_and_zscores$avg_pred
matched_column_names <- summarize_and_save_results(z_scores, residual_all, AD_ROI, numerical_cols, resultsDir, "GAM")

compute_correlation_and_r_squared(z_scores, AD_ROI, sd_pop, data, average_predictions, resultsDir, "GAM", matched_column_names)
outlier_thresh <- 7
generate_plots_gam(gam_models, data_healthy = data_train, average_predictions, AD_ROI=AD_ROI, resultsDir=resultsDir, suffix="GAM", data_patients = data_patients, sd_pop=sd_pop)

# GAMLSS

residuals_and_zscores <- calculate_residuals_and_zscores(gamlss_models, data_test, mydata, AD_ROI, "GAMLSS")
compute_statistics(models = gamlss_models, data_te = data_test, suffix = "GAMLSS", experiment = "healthy", resultsDir = resultsDir)
compute_statistics(models = gamlss_models, data_te = mydata, suffix = "GAMLSS", experiment = "all", resultsDir = resultsDir)
z_scores <- residuals_and_zscores$z_scores
residual_all <- residuals_and_zscores$residual_all
numerical_cols <- names(z_scores)[-(1:2)]
average_predictions <- residuals_and_zscores$avg_pred
matched_column_names <- summarize_and_save_results(z_scores, residual_all, AD_ROI, numerical_cols, resultsDir, "GAMLSS")

compute_correlation_and_r_squared(z_scores, AD_ROI, residuals_and_zscores$sd_pop, data, average_predictions, resultsDir, "GAMLSS", matched_column_names)
generate_plots_gamlss(gamlss_models, data_healthy = data_train, average_predictions, AD_ROI=AD_ROI, resultsDir=resultsDir, suffix="GAMLSS", data_patients = data_patients)


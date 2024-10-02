# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(xgboost)
library(vroom)
library(rpart)
library(stacks)

setwd("//wsl.localhost/Ubuntu/home/fidgetcase/stat348/BikeShare")

# Read in Data ------------------------------------------------------------


test <- vroom("test.csv")
train <- vroom("train.csv")



# EDA ---------------------------------------------------------------------


dplyr::glimpse(train) 
#skimr::skim(test)
# DataExplorer::plot_correlation(train) 
# DataExplorer::plot_bar(train)

train = subset(train, select = -c(casual, registered))
train$count = log(train$count)

# Recipe  -----------------------------------------------------------------


my_recipe <- recipe(count ~ . , data=train) %>% 
  step_mutate(weather = recode(weather, `4` = 3)) %>% # Recode weather condition '4' to '3' and convert it to a factor
  step_mutate(weather = as.factor(weather)) %>%
  step_mutate(
    datetime_hour = hour(datetime),
    datetime_month = month(datetime),
    datetime_wday = wday(datetime),
    datetime_year = year(datetime)) %>%
  step_mutate(hour_sin = sin(2 * pi * datetime_hour / 24), # Create cyclical features for hour (since it repeats daily)
              hour_cos = cos(2 * pi * datetime_hour / 24)) %>%
  step_mutate(season = as.factor(season),  # Convert season, workingday, holiday to factors
              workingday = as.factor(workingday),
              holiday = as.factor(holiday)) %>%
  step_interact(terms = ~ workingday:datetime_hour + season:weather + holiday:weather) %>% # Interactions between categorical variables (e.g., workingday and hour)
  step_dummy(all_nominal_predictors()) %>% # Convert all nominal predictors into dummy variables
  step_normalize(all_numeric_predictors()) %>% # Normalize all numeric predictors (except count)
  step_rm(datetime)

prepped_recipe <- prep(my_recipe) # Sets up the preprocessing 
bake(prepped_recipe, new_data= train)


# Models ------------------------------------------------------------------


## Linear ------------------------------------------------------------------

my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>% # Regression just means quantitative response
  fit(formula=count ~ temp + workingday + windspeed, data=train)

lin_reg <-
linear_reg() %>%
set_engine("lm")


## XGboost mod -------------------------------------------------------------

xgb_spec <- boost_tree(
  trees = 10,            # Number of trees
  tree_depth = tune(),      # Tune the depth of trees
  learn_rate = tune(),      # Learning rate for boosting
  loss_reduction = tune(),  # Minimum loss reduction for further splits
  sample_size = tune(),     # Proportion of data to use for fitting
  mtry = tune(),            # Number of features to sample
  min_n = tune()            # Minimum data points in a node
) %>%
  set_engine("xgboost") %>% # Specify the xgboost engine
  set_mode("regression")    # Set it to a regression problem

## Decision Tree  ------------------------------------------------------------

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=10) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")



# Grids ----------------------------------------------------------------

folds <- vfold_cv(train, v = 5)  # 5-fold cross-validation
# Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples()


## XGBoost Grid ---------------------------------------------------------

xgb_grid <- grid_regular(
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  sample_size = sample_prop(),
  mtry(range= c(1, 10)),
  min_n(),
  levels = 5  # Number of random combinations to try
)

## Random Forest Grid ---------------------------------------------------

grid_of_tuning_params <- grid_regular(mtry(range= c(1, 10)),
                                      min_n(),
                                      levels = 5) ## L^2 total tuning possibilities
# Workflow --------------------------------------------------------------

## XGBoost workflow ------------------------------------------------------
# Workflow: Combine recipe and model
xgb_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%      # Add your existing recipe
  add_model(xgb_spec)            # Add the XGBoost model


## RF workflow ----------------------------------------------------------

randf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)


## Linear Workflow ---------------------------------------------------------

lin_reg_wf <-
  workflow() %>%
  add_model(lin_reg) %>%
  add_recipe(my_recipe)

# Tuning ------------------------------------------------------------------


## XG Boost Tune -----------------------------------------------------------

# Tune the hyperparameters using cross-validation
xgb_tune_results <- tune_grid(
  xgb_workflow,
  resamples = folds,
  grid = xgb_grid,
  metrics = metric_set(rmse),    # Use RMSE to evaluate model performance
  control = tunedModel)


 ## RF Tune -----------------------------------------------------------------

CV_results <- randf_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae),
            control = untunedModel) #Or leave metrics NULL


## Linear Model Tune -------------------------------------------------------

my_linear_model <-
  fit_resamples(
    lin_reg_wf,
    resamples = folds,
    metrics = metric_set(rmse, mae, rsq),
    control = tunedModel)


# Best Models -------------------------------------------------------------


## XG Boost Best -----------------------------------------------------------

# Select the best hyperparameters
best_xgb <- xgb_tune_results |> 
  select_best(metric = "rmse")

# RF Best -----------------------------------------------------------------

best_rf <- CV_results %>%
  select_best(metric="rmse")


# Final -------------------------------------------------------------------


## XG Boost Final ----------------------------------------------------------

# Finalize the workflow with the best hyperparameters
finalxg_wf <- finalize_workflow(
  xgb_workflow,
  best_xgb) %>%
  fit(data = train)

## RF Final ----------------------------------------------------------------

finalrf_wf <-
  randf_wf %>%
  finalize_workflow(best_rf) %>%
  fit(data=train)


# Stack -------------------------------------------------------------------

model_stack <- stacks() %>%
  # Add the tuned models to the stack
  add_candidates(xgb_tune_results) %>%
  add_candidates(CV_results) %>%
  add_candidates(my_linear_model)

# Blend predictions (fit the meta-learner, typically a linear regression)
model_stack <- blend_predictions(model_stack)

# Fit the stacked model on the training data
final_model_stack <- fit_members(model_stack)


# Make predictions on the test set
test_predictions <- final_model_stack %>%
  predict(test)

# Kaggle Submission -------------------------------------------------------


kaggle_submission <- test_predictions %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count), 
         count = exp(count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file= "kagglecomp.csv", delim=",") 
#change the file name to the git hub repository
 
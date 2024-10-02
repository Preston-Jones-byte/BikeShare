
# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(xgboost)
library(vroom)
library(rpart)
library(dials)

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
    datetime_year = year(datetime)
  ) %>%
  step_mutate(hour_sin = sin(2 * pi * datetime_hour / 24), # Create cyclical features for hour (since it repeats daily)
              hour_cos = cos(2 * pi * datetime_hour / 24)) %>%
  step_mutate(season = as.factor(season),  # Convert season, workingday, holiday to factors
              workingday = as.factor(workingday),
              holiday = as.factor(holiday)) %>%
  step_mutate(datetime_year, features = datetime_year) %>%
  step_interact(terms = ~ workingday:datetime_hour + season:weather + holiday:weather) %>% # Interactions between categorical variables (e.g., workingday and hour)
  step_dummy(all_nominal_predictors()) %>% # Convert all nominal predictors into dummy variables
  step_normalize(all_numeric_predictors()) %>% # Normalize all numeric predictors (except count)
  step_rm(datetime)
  
prepped_recipe <- prep(my_recipe) # Sets up the preprocessing 
bake(prepped_recipe, new_data= train)




# Random Forest  ------------------------------------------------------------

# my_mod <- rand_forest(mtry = tune(),
#                       min_n=tune(),
#                       trees=1000) %>% #Type of model
#   set_engine("ranger") %>% # What R function to use
#   set_mode("regression")
# 
# ## Set Workflow 
# randf_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(my_mod)
# 
# ## Grid of values to tune over
# grid_of_tuning_params <- grid_regular(mtry(range= c(1, 10)),
#                                       min_n(),
#                                       levels = 5) ## L^2 total tuning possibilities
# 
# ## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)
# 
# 
# ## Run the CV
# CV_results <- randf_wf %>%
#           tune_grid(resamples=folds,
#           grid=grid_of_tuning_params,
#           metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL
# 
# 
# 
# ## Find Best Tuning Parameters
# bestTune <- CV_results %>%
#   select_best(metric="rmse")
# 
# ## Finalize the Workflow & fit it
# finalrf_wf <-
#   randf_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=train)
# 
# ## Predict
# rf_preds <- finalrf_wf %>%
#   predict(new_data = test)


# Bart --------------------------------------------------------------------

bart_mod <- bart(
  mode = "regression",
  engine = "dbarts",
  trees = 1000,
  prior_terminal_node_coef = tune(),
  prior_terminal_node_expo = tune(),
  prior_outcome_range = tune())

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_mod)

coef_param <- prior_terminal_node_coef() %>%
  update(range = c(0, 0.95))

bart_grid <- grid_regular(coef_param(),
                          prior_terminal_node_expo(),
                          prior_outcome_range(),
                          levels = 5) ## L^2 total tuning possibilities

bart_tuning <- bart_wf %>%
          tune_grid(resamples=folds,
          grid=bart_grid,
          metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL

bart_tuned <- bart_tuning %>%
  select_best(metric="rmse") 


finalbart_wf <- bart_wf %>%
  finalize_workflow(bart_tuned)
  fit(data = train)

bart_preds <- finalbart_wf %>%
  predict(new_data = test)



# Kaggle_submission -------------------------------------------------------


kaggle_submission <- bart_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables  
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count), 
         count = exp(count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file= "kagglesubmission.csv", delim=",") 
              #change the file name to the git hub repository

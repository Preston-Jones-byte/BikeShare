
# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(vroom)
library(rpart)

setwd("//wsl.localhost/Ubuntu/home/fidgetcase/stat348/BikeShare")

# Read in Data ------------------------------------------------------------


test <- vroom("test.csv")
train <- vroom("train.csv")



# EDA ---------------------------------------------------------------------


dplyr::glimpse(train) 
#skimr::skim(test)
# DataExplorer::plot_correlation(train) 
# DataExplorer::plot_bar(train)

## Question 1 -------------------------------------------------------------

train = subset(train, select = -c(casual, registered))
train$count = log(train$count)

# Recipe  -----------------------------------------------------------------

## Question 2 -------------------------------------------------------------

my_recipe <- recipe(count ~ . , data=train) %>% # Set model formula and dataset
  step_mutate(weather = recode(weather, `4` = 3)) %>% # Recodes 4 to 3
  step_mutate(weather = as.factor(weather)) %>% # Weather becomes a factor
  #step_mutate(hour = hour(datetime)) %>%
  step_time(datetime, features = c("hour")) %>% # We have a new time variable
  step_mutate(season = as.factor(season)) %>% # Season is a factor
  step_mutate(workingday = as.factor(workingday)) %>%
  step_mutate(datetime_hour = as.factor(datetime_hour)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_rm(datetime)
  
  
prepped_recipe <- prep(my_recipe) # Sets up the preprocessing 
bake(prepped_recipe, new_data= train)




# Decision Tree  ------------------------------------------------------------

my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) |> 
  set_engine("rpart") |> 
  set_mode("regression")

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(tree_depth(),
                                      min_n(),
                                      cost_complexity(),
                                      levels = 10) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 10, repeats=1)


## Run the CV
CV_results <- preg_wf %>%
          tune_grid(resamples=folds,
          grid=grid_of_tuning_params,
          metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL



## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric="rmse")

## Finalize the Workflow & fit it
final_wf <-
  preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
x <- final_wf %>%
  predict(new_data = test)

# Kaggle_submission -------------------------------------------------------


kaggle_submission <- x %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count), 
         count = exp(count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file= "regressiontreehw9.csv", delim=",") 
              #change the file name to the git hub repository

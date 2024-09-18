
# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(vroom)

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
  step_dummy(all_nominal_predictors()) 
  
  
prepped_recipe <- prep(my_recipe) # Sets up the preprocessing 
bake(prepped_recipe, new_data= train)


# Example Code ------------------------------------------------------------




# Workflow ----------------------------------------------------------------
## Question 3 -------------------------------------------------------------

## Define a Recipe as before
bike_recipe <- my_recipe

## Define a Model
lin_model <- linear_reg() %>%
set_engine("lm") %>%
set_mode("regression")

## Combine into a Workflow and fit
bike_workflow <- workflow() %>%
add_recipe(bike_recipe) %>%
add_model(lin_model) %>%
fit(data=train)

## Run all the steps on test data
lin_preds <- exp(predict(bike_workflow, new_data = test))



# Kaggle_submission -------------------------------------------------------


kaggle_submission <- lin_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file= "Workflowhw6.csv", delim=",") 
              #change the file name to the git hub repository

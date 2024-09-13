library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(patchwork)
setwd("//wsl.localhost/Ubuntu/home/fidgetcase/stat348/BikeShare")

test <- vroom("test.csv")
train <- vroom("train.csv")

dplyr::glimpse(train) 
skimr::skim(test)
DataExplorer::plot_correlation(train) 
DataExplorer::plot_bar(train)


plot1 <- ggplot(data = train, aes(x = atemp, y = temp)) +
  geom_point()

plot2 <- ggplot(data = train, aes(x = weather, y = count)) +
  geom_col() 

plot3 <- ggplot(data = train, aes(x = holiday)) +
  geom_bar()

plot4 <- ggplot(data = train, aes(x = atemp, y = weather)) +
  geom_col() 

(plot1 + plot2) / (plot3 + plot4)

my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>% # Regression just means quantitative response
  fit(formula=count ~ temp + workingday + windspeed, data=train)

## Generate Predictions Using Linear Model9
bike_predictions <- predict(my_linear_model,
                            new_data=test)

kaggle_submission <- bike_predictions %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file= "./LinearPreds.csv", delim=",") 
              #change the file name to the git hub repository

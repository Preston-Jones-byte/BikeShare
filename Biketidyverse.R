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

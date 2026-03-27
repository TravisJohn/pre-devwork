library(tidyverse)
library(ggplot2)
tasks <- c("Literature Review", "Data Exploration/Analysis", "Modelling", "Model Evaluation", "Report", "Presentation")
dfr <- data.frame(
  name        = factor(tasks, levels = rev(tasks)),
  start.date  = as.Date(c("2024-08-19", "2024-08-26", "2024-09-09", "2024-09-16","2024-09-23","2024-09-30")),
  end.date    = as.Date(c("2024-09-08", "2024-09-15", "2024-09-22", "2024-09-29","2024-10-05","2024-10-08"))
)
mdfr <- dfr %>%
  pivot_longer(cols = c("start.date", "end.date"), 
               names_to = "variable", 
               values_to = "value")

event_dates <- as.Date(c("2024-09-03", "2024-09-24", "2024-10-05","2024-10-08"))
event_labels <- c("A1", "A2", "A3", "A4")

ggplot(mdfr, aes(value, name)) + 
  geom_line(size = 6) +
  geom_vline(xintercept = event_dates, linetype = "dashed", color = "red") +
  annotate("text", 
           x = event_dates, 
           y = 0.7, 
           label = event_labels, 
           angle = 90, 
           vjust = 1.5, 
           hjust = 1, 
           size = 3.5,  # Matches the size of axis text in theme_minimal()
           family = "sans",  # Matches the default font family
           color = "black") +  # Matches the default text color
  ggtitle("ZZSC9020 Group11 Gantt Chart") +
  xlab(NULL) + 
  ylab(NULL) +
  theme_minimal()



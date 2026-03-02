##R Code for PCA and FA, HW 10 (LH revised, 4/6/22, to add tests of assumptions, etc.)
# Install.packages("parallel") #Install any needed packages
# Load needed libraries (load relevant ones every session you run R)
library(psych); library(dplyr); library(car); library(readxl);
library(GPArotation);
library(lsr);
library(dplyr)
library(psych)
library(lsmeans)
library(car)
library(lawstat)
library(psychometric)
##########################################################################################
# Analysis for the duration.
##########################################################################################
getwd();
# /Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/PD/Participant1/peak_amplitudes.csv
setwd("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/GitQ2Bheave/q2behave_data_analysis/");

activity_1_duration <- read.csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Q2Behave/CodeOutputs/DurationOfEachBlob/duration.csv");
unique_values <- unique(activity_1_duration$newFeatureName)
value_counts <- table(activity_1_duration$newFeatureName)

# ADHD ----- P7, P9,P11,P14,P16,P24
# Neurotypical --- P8, P10,P12,P13,P15,P17,P18

# List of participants to subset
#participants <- c("P7", "P9","P11","P14","P16","P24","P8", "P10","P12","P13","P15","P17") # "P18" 6 total in ADHD and 6 In neurotypical.
# Right hand only.
# unique(activity_1_duration_right$PID)
# "P7"  "P24" "P12" "P8"  "P6"  "P15" "P9"  "P18" "P14" "P13" "P23" "P21" "P16" "P20" "P11" "P22" "P5"  "P10"
activity_1_duration_right <- subset(activity_1_duration, hand_list == "Right")
# Remove no hyperactive durations
activity_1_duration_right_hy = subset(activity_1_duration_right, newFeatureName != "NH") 
# Subset data based on the list of participants
#subset_data <- activity_1_duration_right_hy #activity_1_duration_right_hy[activity_1_duration_right_hy$PID %in% participants,]
subset_data <- subset(subset_data, newFeatureName != "SW") 

# Define a vector of participant IDs to assign values
target_participants <- c("P7", "P9","P11","P14","P16","P24")
# Assign values of 0 or 1 based on participant ID
subset_data$Assignment <- ifelse(subset_data$PID %in% target_participants, 1, 0)

subset_data_MC <- subset(subset_data, newFeatureName == "FT") 

# H0: The mean duration of hyperactive behavior is the same in ADHD and Neurotypical particpants. 
t_test_result <- t.test(Duration~Assignment, data=subset_data)
print(t_test_result)


value_counts_fidget_behavior <- table(subset_data$PID)


df_counts <- as.data.frame(value_counts_fidget_behavior)

# Rename the columns
names(df_counts) <- c("PID", "Frequency")

df_counts$Assignment <- ifelse(df_counts$PID %in% target_participants, 1, 0)


# Compute t-test
res <- t.test(Frequency~Assignment,paired = TRUE, data=df_counts, var.equal = FALSE)
print(res)






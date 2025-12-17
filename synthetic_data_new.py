import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaEngine

# Load cleaned real dataset
df = pd.read_csv("C:/Users/pc/Downloads/INT234 project/cleaned_dataset.csv")

# Tell SDV which columns MUST be integers
integer_columns = [
    "Age", "Daily_Study_Hours", "Weekly_Revision_Days",
    "Study_Hours_Before_Exam", "Number_of_Subjects",
    "Class_Attendance_Percentage", "Assignment_Quality_Rating",
    "Exercise_Days_Per_Week", "Daily_Commute_Time_Minutes",
    "Phone_Usage_During_Study_Minutes", "Caffeine_Intake_Cups"
]

# Binary columns
binary_columns = ["Group_Study", "Coaching_or_Tuition",
                  "Part_Time_Job", "Uses_AI_Tools",
                  "Uses_Handwritten_Notes"]

# Train SDV model
engine = GaussianCopulaEngine(df)
engine.fit(df)

# Generate synthetic data
synthetic = engine.generate(1000)

# ---- FIX TYPES ----

# Round integer columns
for col in integer_columns:
    if col in synthetic.columns:
        synthetic[col] = synthetic[col].round().astype(int)

# Force binary columns to 0/1
for col in binary_columns:
    if col in synthetic.columns:
        synthetic[col] = np.where(synthetic[col] > 0.5, 1, 0).astype(int)

# Clip unrealistic values
synthetic["Age"] = synthetic["Age"].clip(17, 40)  
synthetic["Class_Attendance_Percentage"] = synthetic["Class_Attendance_Percentage"].clip(0, 100)
synthetic["Final_Score_Percentage"] = synthetic["Final_Score_Percentage"].clip(0, 100)
synthetic["Daily_Study_Hours"] = synthetic["Daily_Study_Hours"].clip(0, 16)
synthetic["Daily_Commute_Time_Minutes"] = synthetic["Daily_Commute_Time_Minutes"].clip(0, 180)

# Save
synthetic.to_csv("synthetic_fixed.csv", index=False)

print("DONE! Generated synthetic_fixed.csv with correct integer formatting.")
print(synthetic.head(10))

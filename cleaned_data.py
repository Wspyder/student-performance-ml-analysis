import pandas as pd
import numpy as np
import os

INPUT = "C:/Users/pc/Downloads/INT234 project/INT234 projectDataset.csv"
OUTPUT = "cleaned_dataset.csv"        

#1) load data
df = pd.read_csv(INPUT)
print("Loaded rows:", len(df))

#2) strip spaces from string columns and make simple case fixes
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.strip()

#3) Fix Final_Score_Percentage safely
if 'Final_Score_Percentage' in df.columns:
    # remove percent sign if present
    df['Final_Score_Percentage'] = df['Final_Score_Percentage'].astype(str).str.replace('%','', regex=False).str.strip()
    # convert to numeric; invalid -> NaN
    df['Final_Score_Percentage'] = pd.to_numeric(df['Final_Score_Percentage'], errors='coerce')
    # if tiny fractions like 0.74 appear, multiply by 100
    df['Final_Score_Percentage'] = df['Final_Score_Percentage'].apply(
        lambda x: x*100 if (pd.notnull(x) and x <= 1) else x
    )
    df['Final_Score_Percentage'] = df['Final_Score_Percentage'].round(2)
else:
    raise KeyError("Final_Score_Percentage column not found. Check your CSV header.")

#4) Standardize Program_Stream (simple mapping)
if 'Program_Stream' in df.columns:
    df['Program_Stream'] = df['Program_Stream'].astype(str).str.upper().str.replace('.', '', regex=False).str.replace('  ',' ', regex=False)
    prog_map = {
        'CSE':'CSE','C SE':'CSE','CSE ':'CSE','CSE\t':'CSE',
        'B.DES':'B.DES','BDES':'B.DES','B.DES':'B.DES',
        'BBA':'BBA','BA':'BA','BA LLB':'BA LLB','BA.LLB':'BA LLB','LLB':'LLB',
        'BHMS':'BHMS','MBBS':'MBBS','BSC':'BSC','BCA':'BCA','DROP':'DROPOUT','DROPOUT':'DROPOUT'
    }
    df['Program_Stream'] = df['Program_Stream'].map(lambda x: prog_map.get(x.strip(), x.strip()))

#5) Normalize Gender
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].astype(str).str.strip().str.title()
    df['Gender'] = df['Gender'].replace({
        'Male':'Male','Female':'Female','Non-Binary':'Non-binary','Non-binary':'Non-binary',
        'Prefer Not To Say':'Prefer not to say','Prefer not to say':'Prefer not to say'
    })

#6) Convert Yes/No columns to 0/1
yesno = ["Group_Study","Coaching_or_Tuition","Part_Time_Job","Uses_AI_Tools","Uses_Handwritten_Notes"]
for c in yesno:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.lower().map({'yes':1,'no':0})
        df[c] = df[c].fillna(0).astype(int)

#7) Numeric conversion for candidate columns
numeric_cols = [
    "Age","Daily_Study_Hours","Weekly_Revision_Days","Study_Hours_Before_Exam",
    "Number_of_Subjects","Class_Attendance_Percentage","Assignment_Quality_Rating",
    "Average_Sleep_Hours","Caffeine_Intake_Cups","Daily_Screen_Time_Hours",
    "Exercise_Days_Per_Week","Daily_Commute_Time_Minutes","Stress_Level",
    "Motivation_Level","Concentration_Ability","Study_Environment_Quality",
    "Phone_Usage_During_Study_Minutes"
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

#8) Handle missing values simply
# numeric -> median; categorical -> mode
for c in numeric_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].fillna(df[c].mode().iloc[0])

#9) Mild outlier handling
if 'Daily_Study_Hours' in df.columns:
    df['Daily_Study_Hours'] = df['Daily_Study_Hours'].clip(lower=0, upper=16)
if 'Average_Sleep_Hours' in df.columns:
    df['Average_Sleep_Hours'] = df['Average_Sleep_Hours'].clip(lower=0, upper=16)
if 'Daily_Commute_Time_Minutes' in df.columns:
    df['Daily_Commute_Time_Minutes'] = df['Daily_Commute_Time_Minutes'].clip(lower=0, upper=600)

#10) Quick checks
print("Final_Score: min/max/mean ->", df['Final_Score_Percentage'].min(), df['Final_Score_Percentage'].max(), round(df['Final_Score_Percentage'].mean(),2))
if 'Program_Stream' in df.columns:
    print("Program_Stream value counts (top 10):")
    print(df['Program_Stream'].value_counts().head(10))
if 'Gender' in df.columns:
    print("Gender value counts:")
    print(df['Gender'].value_counts())

#11) Save cleaned file
df.to_csv(OUTPUT, index=False)
print("Saved cleaned data to:", OUTPUT)

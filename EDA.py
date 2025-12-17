import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
OUTDIR = "eda_outputs"
os.makedirs(OUTDIR, exist_ok=True)

#load cleaned synthetic dataset
CSV_PATH = "C:/Users/pc/Downloads/INT234 project/synth_outputs/synthetic_final_cleaned.csv"
df = pd.read_csv(CSV_PATH)

print("Dataset Shape:", df.shape)
print("Columns:\n", df.columns.tolist())
print("\nSummary statistics (numeric):\n", df.describe())

#numeric-only dataframe for correlations and numeric plots
num_df = df.select_dtypes(include=[np.number]).copy()
print("\nNumeric columns used for correlation:", num_df.columns.tolist())

#1) Distribution of Final Score
plt.figure(figsize=(7,5))
sns.histplot(df["Final_Score_Percentage"], kde=True, bins=20)
plt.title("Distribution of Final Score (%)")
plt.xlabel("Final Score (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "dist_final_score.png"))
plt.show()

#2) Scatter: Daily_Study_Hours vs Final Score
if "Daily_Study_Hours" in num_df.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=num_df["Daily_Study_Hours"], y=num_df["Final_Score_Percentage"], alpha=0.6)
    plt.title("Daily Study Hours vs Final Score")
    plt.xlabel("Daily Study Hours")
    plt.ylabel("Final Score (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "scatter_study_vs_score.png"))
    plt.show()
else:
    print("Skipping scatter plot: Daily_Study_Hours not numeric or missing.")

#3) Scatter: Average_Sleep_Hours vs Final Score 
if "Average_Sleep_Hours" in num_df.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=num_df["Average_Sleep_Hours"], y=num_df["Final_Score_Percentage"], alpha=0.6)
    plt.title("Average Sleep Hours vs Final Score")
    plt.xlabel("Average Sleep Hours")
    plt.ylabel("Final Score (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "scatter_sleep_vs_score.png"))
    plt.show()

#4) Scatter: Class_Attendance_Percentage vs Final Score
if "Class_Attendance_Percentage" in num_df.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=num_df["Class_Attendance_Percentage"], y=num_df["Final_Score_Percentage"], alpha=0.6)
    plt.title("Attendance vs Final Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Score (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "scatter_attendance_vs_score.png"))
    plt.show()

#5) Correlation heatmap
    corr = num_df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation matrix (numeric features)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "corr_heatmap_numeric.png"))
plt.show()

#6) Highest correlations with Final_Score_Percentage
if "Final_Score_Percentage" in corr.columns:
    corr_with_target = corr["Final_Score_Percentage"].drop("Final_Score_Percentage").sort_values(ascending=False)
    print("\nTop features positively correlated with Final Score:\n", corr_with_target.head(10))
    print("\nTop features negatively correlated with Final Score:\n", corr_with_target.tail(10))
    corr_with_target.head(10).to_csv(os.path.join(OUTDIR, "top_pos_correlations.csv"))
else:
    print("Final_Score_Percentage not numeric, cannot compute correlation list.")

#7) Categorical counts: Preferred_Study_Time
if "Preferred_Study_Time" in df.columns:
    plt.figure(figsize=(7,4))
    order = df["Preferred_Study_Time"].value_counts().index
    sns.countplot(y="Preferred_Study_Time", data=df, order=order)
    plt.title("Preferred Study Time (counts)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "preferred_study_time_counts.png"))
    plt.show()

print("\nEDA completed. Plots and CSVs saved in:", OUTDIR)

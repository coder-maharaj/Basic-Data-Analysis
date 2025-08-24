# Titanic Dataset - Basic Data Analysis
# Author: Nirvana

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (from seaborn for simplicity, no CSV needed)
titanic = sns.load_dataset("titanic")

# Preview data
print("Dataset Preview:")
print(titanic.head())

# Basic info
print("\nDataset Info:")
print(titanic.info())

# Check missing values
print("\nMissing Values:")
print(titanic.isnull().sum())

# Survival count
print("\nSurvival Count:")
print(titanic['survived'].value_counts())

# Survival rate by gender
print("\nSurvival Rate by Gender:")
print(titanic.groupby("sex")["survived"].mean())

# Visualization 1: Survival by gender
sns.countplot(x="sex", hue="survived", data=titanic)
plt.title("Survival by Gender")
plt.show()

# Visualization 2: Survival by passenger class
sns.countplot(x="class", hue="survived", data=titanic)
plt.title("Survival by Passenger Class")
plt.show()

# Visualization 3: Age distribution of survivors
sns.histplot(data=titanic, x="age", hue="survived", bins=30, kde=True)
plt.title("Age Distribution of Survivors")
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
df = pd.read_csv('dataset\\cleveland.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())
print(df.info())
df.to_csv('dataset\\cleaned_cleveland.csv')

#Variation of Age for each target class
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='age', hue='target', palette='Set2')

plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Variation of Age for each target class')
plt.xticks(rotation=90)
plt.legend(title='Target')
plt.savefig('graph\\variation_age_target.png', dpi=300, bbox_inches='tight')


#Distribution of age vs sex with the target class
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='sex', y='age', hue='target', palette='muted')

plt.xlabel('Sex')
plt.ylabel('Age')
plt.title('Distribution of age vs sex with the target class')
plt.legend(title='Target')
plt.savefig('graph\\distribution_age_sex_target.png', dpi=300, bbox_inches='tight')
                             
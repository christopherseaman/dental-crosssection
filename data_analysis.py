import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the dataset and preprocess the data
control_data = pd.read_excel('control_data.xlsx', header=[0, 1])
test_data = pd.read_excel('test_data.xlsx', header=[0, 1])

# Perform statistical analysis on the data
control_scores = control_data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').mean(axis=1)
test_scores = test_data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').mean(axis=1)

t_stat, p_value = stats.ttest_ind(control_scores, test_scores)

# Generate visualizations to represent the analysis results
plt.hist(control_scores, alpha=0.5, label='Control')
plt.hist(test_scores, alpha=0.5, label='Test')
plt.legend(loc='upper right')
plt.title('Score Distribution')
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.show()

print(f'T-statistic: {t_stat}, P-value: {p_value}')

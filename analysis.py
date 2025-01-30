import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Import data
print("\n=== Step 1: Data Import ===\n")

# 1.1 Import test results
test_results = pd.read_csv('data/test_results.tsv', sep='\t')
print("Test Results Shape:", test_results.shape)
print("\nTest Results Columns:")
print(test_results.columns.tolist())
print("\nTest Results Preview:")
print(test_results.head())

# 1.2 Import question grouping
# Skip the header row and provide column names explicitly
question_groups = pd.read_csv('data/question_groups.tsv', sep='\t', skiprows=1, names=['Question', 'QuestionGroup'])
print("\nQuestion Groups Shape:", question_groups.shape)
print("\nQuestion Groups Preview:")
print(question_groups.head())

# 1.3 Import post-test survey
posttest_survey = pd.read_csv('data/posttest_survey.tsv', sep='\t')
print("\nPost-test Survey Shape:", posttest_survey.shape)
print("\nPost-test Survey Columns:")
print(posttest_survey.columns.tolist())
print("\nPost-test Survey Preview:")
print(posttest_survey.head())

# Step 2: Data Quality & Preprocessing
print("\n=== Step 2: Data Quality & Preprocessing ===\n")

# 2.1 Check for missing values
print("Missing Values in Test Results:")
print(test_results.isnull().sum())
print("\nMissing Values in Question Groups:")
print(question_groups.isnull().sum())
print("\nMissing Values in Post-test Survey:")
print(posttest_survey.isnull().sum())

# 2.2 Check for duplicates
print("\nDuplicate Rows:")
print("Test Results:", test_results.duplicated().sum())
print("Question Groups:", question_groups.duplicated().sum())
print("Post-test Survey:", posttest_survey.duplicated().sum())

# 2.3 Convert test duration to seconds
def duration_to_seconds(duration_str):
    if 'm' not in duration_str:
        return None
    
    # Extract minutes and seconds
    parts = duration_str.split('m')
    minutes = int(parts[0])
    seconds = 0
    if 's' in parts[1]:
        seconds = int(parts[1].split('s')[0].strip())
    
    return minutes * 60 + seconds

test_results['DurationSeconds'] = test_results['TestDuration'].apply(duration_to_seconds)

# 2.4 Convert Likert scales to numeric values
likert_mapping = {
    'Strongly disagree': 1,
    'Somewhat disagree': 2,
    'Neither agree nor disagree': 3,
    'Somewhat agree': 4,
    'Strongly agree': 5
}

for col in ['Q4_1', 'Q4_2', 'Q4_3']:
    posttest_survey[f'{col}_numeric'] = posttest_survey[col].map(likert_mapping)

# 2.5 Convert Score to numeric (remove % and convert to float)
test_results['ScoreNumeric'] = test_results['Score'].str.rstrip('%').astype(float) / 100

# 2.6 Outlier detection for test durations using IQR method
Q1 = test_results['DurationSeconds'].quantile(0.25)
Q3 = test_results['DurationSeconds'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = test_results[
    (test_results['DurationSeconds'] < lower_bound) |
    (test_results['DurationSeconds'] > upper_bound)
]

print("\nTest Duration Statistics (in seconds):")
print(test_results['DurationSeconds'].describe())
print("\nNumber of Duration Outliers:", len(outliers))
if len(outliers) > 0:
    print("\nOutlier Details:")
    print(outliers[['ID', 'TestDuration', 'DurationSeconds', 'Score']])

# 2.7 Clean the datasets
# Remove duplicates from question groups
question_groups = question_groups.drop_duplicates()

# Create clean versions of dataframes
test_results_clean = test_results.copy()
posttest_survey_clean = posttest_survey.dropna(subset=['Cohort'])  # Remove rows with missing cohort

# Print cohort sizes
print("\nCohort sizes in test_results_clean:")
print(test_results_clean['Cohort'].value_counts())

print("\nCohort sizes in posttest_survey_clean:")
print(posttest_survey_clean['Cohort'].value_counts())

# Check unique values in Cohort column
print("\nUnique values in test_results_clean['Cohort']:")
print(test_results_clean['Cohort'].unique())

print("\nUnique values in posttest_survey_clean['Cohort']:")
print(posttest_survey_clean['Cohort'].unique())

# Step 3: Descriptive Statistics & Visualizations
print("\n=== Step 3: Descriptive Statistics & Visualizations ===\n")

# 3.1 Results: scores and durations
print("Score Statistics by Cohort:")
score_stats = test_results_clean.groupby('Cohort')['ScoreNumeric'].agg([
    'count', 'mean', 'std', 'min', 'max',
    lambda x: x.quantile(0.25).round(3),
    lambda x: x.quantile(0.75).round(3)
]).round(3)
score_stats.columns = ['count', 'mean', 'std', 'min', 'max', 'Q1', 'Q3']
print(score_stats)

print("\nDuration Statistics by Cohort (in minutes):")
duration_stats = test_results_clean.groupby('Cohort')['DurationSeconds'].agg([
    'count', 'mean', 'std', 'min', 'max',
    lambda x: x.quantile(0.25).round(3),
    lambda x: x.quantile(0.75).round(3)
]).round(3)
duration_stats.columns = ['count', 'mean', 'std', 'min', 'max', 'Q1', 'Q3']
duration_stats[['mean', 'std', 'min', 'max', 'Q1', 'Q3']] = \
    duration_stats[['mean', 'std', 'min', 'max', 'Q1', 'Q3']] / 60  # Convert to minutes
print(duration_stats)

# 3.2 Post-test Survey Likert Responses
print("\nLikert Scale Response Statistics:")
for col in ['Q4_1', 'Q4_2', 'Q4_3']:
    print(f"\n{col} Response Distribution:")
    print(posttest_survey_clean[col].value_counts().sort_index())
    print(f"\nNumeric Statistics for {col}:")
    print(posttest_survey_clean[f'{col}_numeric'].describe().round(3))

# Correlation Analysis
print("\nCorrelation Analysis:")

# Test normality of score and duration
score_normal = stats.shapiro(test_results_clean['ScoreNumeric'])[1] > 0.05
duration_normal = stats.shapiro(test_results_clean['DurationSeconds'])[1] > 0.05

# Choose correlation test based on normality
if score_normal and duration_normal:
    correlation, p_value = stats.pearsonr(test_results_clean['ScoreNumeric'], 
                                        test_results_clean['DurationSeconds'])
    print("Both variables are normal, using Pearson correlation:")
else:
    correlation, p_value = stats.spearmanr(test_results_clean['ScoreNumeric'], 
                                         test_results_clean['DurationSeconds'])
    print("One or both variables non-normal, using Spearman correlation:")

print(f"Correlation coefficient: {correlation:.3f}")
print(f"P-value: {p_value:.3f}")

# 3.3 Visualizations
print("\nGenerating visualizations...")

# Set style
sns.set_theme(style="whitegrid")

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Score Distribution by Cohort
plt.figure(figsize=(10, 6))
for cohort in test_results_clean['Cohort'].unique():
    scores = test_results_clean[test_results_clean['Cohort'] == cohort]['ScoreNumeric']
    sns.kdeplot(data=scores, label=cohort, fill=True, alpha=0.3)
plt.title('Score Distribution by Cohort')
plt.xlabel('Score (%)')
plt.ylabel('Density')
plt.legend()
plt.savefig('figures/score_distribution.png')
plt.close()

# Duration Distribution by Cohort
plt.figure(figsize=(10, 6))
for cohort in test_results_clean['Cohort'].unique():
    durations = test_results_clean[test_results_clean['Cohort'] == cohort]['DurationSeconds'] / 60  # Convert to minutes
    sns.kdeplot(data=durations, label=cohort, fill=True, alpha=0.3)
plt.title('Test Duration Distribution by Cohort')
plt.xlabel('Duration (minutes)')
plt.ylabel('Density')
plt.legend()
plt.savefig('figures/duration_distribution.png')
plt.close()

# Box plots comparing scores
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cohort', y='ScoreNumeric', data=test_results_clean)
plt.title('Score Distribution Box Plot by Cohort')
plt.xlabel('Cohort')
plt.ylabel('Score (%)')
plt.savefig('figures/score_boxplot.png')
plt.close()

# Box plots comparing durations
plt.figure(figsize=(10, 6))
test_results_clean['DurationMinutes'] = test_results_clean['DurationSeconds'] / 60
sns.boxplot(x='Cohort', y='DurationMinutes', data=test_results_clean)
plt.title('Test Duration Distribution Box Plot by Cohort')
plt.xlabel('Cohort')
plt.ylabel('Duration (minutes)')
plt.savefig('figures/duration_boxplot.png')
plt.close()

# Correlation plot with confidence bands
plt.figure(figsize=(10, 6))
sns.regplot(x='DurationMinutes', y='ScoreNumeric', data=test_results_clean,
           scatter_kws={'alpha':0.5}, line_kws={'color': 'red'},
           ci=95)  # Add 95% confidence interval bands
plt.title('Score vs. Duration with 95% Confidence Bands')
plt.xlabel('Duration (minutes)')
plt.ylabel('Score (%)')
plt.savefig('figures/score_vs_duration.png')
plt.close()

# Improved Likert plot function
def create_likert_plot(data, questions, question_labels, title):
    """Create a diverging stacked bar chart for Likert scale data."""
    # Response categories in order
    categories = ['Strongly disagree', 'Somewhat disagree', 
                 'Neither agree nor disagree', 
                 'Somewhat agree', 'Strongly agree']
    
    # Calculate percentages
    results = []
    for q in questions:
        d = data[q].value_counts(normalize=True) * 100
        d = d.reindex(categories)
        results.append(d)
    
    df = pd.DataFrame(results, index=question_labels)
    
    # Create diverging bars
    plt.figure(figsize=(12, 8))
    
    # Colors for different responses (from negative to positive)
    colors = ['#ca0020', '#f4a582', '#f7f7f7', '#92c5de', '#0571b0']
    
    # Calculate the middle point for neutral responses
    neutral_idx = len(categories) // 2
    
    # Initialize the left positions
    left = np.zeros(len(questions))
    
    # Plot negative responses (from middle to left)
    for idx in range(neutral_idx):
        width = df[categories[idx]]
        plt.barh(df.index, -width, left=-left, color=colors[idx], 
                label=categories[idx])
        left += width
    
    # Reset left position for positive responses
    left = np.zeros(len(questions))
    
    # Plot neutral and positive responses (from middle to right)
    for idx in range(neutral_idx, len(categories)):
        width = df[categories[idx]]
        plt.barh(df.index, width, left=left, color=colors[idx], 
                label=categories[idx])
        left += width
    
    # Customize plot
    plt.axvline(0, color='black', linestyle='-', alpha=0.3)
    plt.title(title, pad=20)
    plt.xlabel('Percentage of Responses')
    
    # Add percentage labels on the bars
    for i in range(len(df.index)):
        cumsum = 0
        for j in range(len(categories)):
            value = df.iloc[i][categories[j]]
            if j < neutral_idx:
                x = -cumsum - value/2
                cumsum += value
            else:
                x = cumsum + value/2
                cumsum += value
            if value >= 5:  # Only show labels for segments >= 5%
                plt.text(x, i, f'{value:.0f}%', 
                        ha='center', va='center')
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              title='Response Scale')
    
    # Set axis limits to ensure symmetry
    max_val = max(abs(plt.xlim()[0]), abs(plt.xlim()[1]))
    plt.xlim(-max_val, max_val)
    
    plt.tight_layout()
    return plt

# Create Likert plots for each cohort
likert_questions = ['Q4_1', 'Q4_2', 'Q4_3']
question_labels = [
    'Prepared for interpreting\nradiological images',
    'Satisfied with presentation\nof anatomical structures',
    'More confident in identifying\nstructures on images'
]

# Map the cohort names from test data to survey data
cohort_mapping = {
    'cross-section': 'Control',
    'diagnostic-images': 'Experimental'
}

# Print response counts for each cohort
print("\nLikert Response Counts by Cohort:")
for test_cohort, survey_cohort in cohort_mapping.items():
    cohort_data = posttest_survey_clean[posttest_survey_clean['Cohort'] == survey_cohort]
    print(f"\n{test_cohort} Group (n={len(cohort_data)}):")
    for q in likert_questions:
        print(f"\n{q} Distribution:")
        print(cohort_data[q].value_counts().sort_index())
    
    # Create cohort-specific plot
    fig = create_likert_plot(cohort_data, likert_questions, 
                            question_labels,
                            f'Student Feedback - {test_cohort} Group (n={len(cohort_data)})')
    plt.savefig(f'figures/likert_responses_{test_cohort}.png', bbox_inches='tight', dpi=300)
    plt.close()

# Create combined plot for comparison
fig = create_likert_plot(posttest_survey_clean, likert_questions, 
                        question_labels,
                        f'Student Feedback - All Groups (n={len(posttest_survey_clean)})')
plt.savefig('figures/likert_responses_combined.png', bbox_inches='tight', dpi=300)
plt.close()

# Step 4: Statistical Analysis
print("\n=== Step 4: Statistical Analysis ===\n")

# 4.1 Power Analysis
from scipy import stats
from statsmodels.stats.power import TTestPower

# Calculate effect size (Cohen's d) for scores
control_scores = test_results_clean[test_results_clean['Cohort'] == 'cross-section']['ScoreNumeric']
diagnostic_scores = test_results_clean[test_results_clean['Cohort'] == 'diagnostic-images']['ScoreNumeric']

cohens_d = (control_scores.mean() - diagnostic_scores.mean()) / np.sqrt(
    ((len(control_scores) - 1) * control_scores.var() + 
     (len(diagnostic_scores) - 1) * diagnostic_scores.var()) / 
    (len(control_scores) + len(diagnostic_scores) - 2))

# Calculate achieved power
analysis = TTestPower()
achieved_power = analysis.power(
    effect_size=abs(cohens_d),
    nobs=len(control_scores) + len(diagnostic_scores),
    alpha=0.05
)

print("\nPower Analysis:")
print(f"Cohen's d effect size: {cohens_d:.3f}")
print(f"Achieved power: {achieved_power:.3f}")

# Calculate minimum detectable effect size
min_effect_size = analysis.solve_power(
    power=0.8,  # Standard target power
    nobs=len(control_scores) + len(diagnostic_scores),
    alpha=0.05
)
print(f"Minimum detectable effect size (80% power): {min_effect_size:.3f}")

# 4.2 Test Score Analysis
print("\nTest Score Analysis:")

# Normality test for each cohort
for cohort in test_results_clean['Cohort'].unique():
    cohort_scores = test_results_clean[test_results_clean['Cohort'] == cohort]['ScoreNumeric']
    stat, p_value = stats.shapiro(cohort_scores)
    print(f"\nShapiro-Wilk test for {cohort}:")
    print(f"Statistic: {stat:.3f}, p-value: {p_value:.3f}")

# Perform t-test or Mann-Whitney U test based on normality
control_scores = test_results_clean[test_results_clean['Cohort'] == 'cross-section']['ScoreNumeric']
diagnostic_scores = test_results_clean[test_results_clean['Cohort'] == 'diagnostic-images']['ScoreNumeric']

# Get normality test results
diagnostic_stat, diagnostic_p = stats.shapiro(diagnostic_scores)
control_stat, control_p = stats.shapiro(control_scores)
both_normal = diagnostic_p > 0.05 and control_p > 0.05

print("\nTest selection for scores:")
print(f"diagnostic-images normality: p = {diagnostic_p:.3f} ({'normal' if diagnostic_p > 0.05 else 'non-normal'})")
print(f"cross-section normality: p = {control_p:.3f} ({'normal' if control_p > 0.05 else 'non-normal'})")
print(f"Using {'t-test' if both_normal else 'Mann-Whitney U test'} based on normality results")

if both_normal:
    # Use t-test for normal distributions
    stat, p_value = stats.ttest_ind(control_scores, diagnostic_scores)
    print("\nBoth groups are normally distributed, using Independent t-test:")
    print(f"t-statistic: {stat:.3f}, p-value: {p_value:.3f}")
    
    # Calculate Cohen's d effect size for t-test
    n1, n2 = len(control_scores), len(diagnostic_scores)
    pooled_std = np.sqrt(((n1 - 1) * control_scores.std()**2 + (n2 - 1) * diagnostic_scores.std()**2) / (n1 + n2 - 2))
    effect_size = abs(control_scores.mean() - diagnostic_scores.mean()) / pooled_std
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
else:
    # Use Mann-Whitney U test for non-normal distributions
    stat, p_value = stats.mannwhitneyu(control_scores, diagnostic_scores, alternative='two-sided')
    print("\nOne or both groups are non-normal, using Mann-Whitney U test:")
    print(f"U-statistic: {stat:.3f}, p-value: {p_value:.3f}")
    
    # Calculate effect size r = Z/√N for Mann-Whitney U test
    n1, n2 = len(control_scores), len(diagnostic_scores)
    z_score = stats.norm.ppf(p_value/2)
    effect_size = abs(z_score / np.sqrt(n1 + n2))
    print(f"Effect size (r): {effect_size:.3f}")

# 4.3 Duration Analysis
print("\nDuration Analysis:")

# Get duration data and test normality
control_durations = test_results_clean[test_results_clean['Cohort'] == 'cross-section']['DurationSeconds']
diagnostic_durations = test_results_clean[test_results_clean['Cohort'] == 'diagnostic-images']['DurationSeconds']

# Normality test for durations
for cohort in test_results_clean['Cohort'].unique():
    cohort_durations = test_results_clean[test_results_clean['Cohort'] == cohort]['DurationSeconds']
    stat, p_value = stats.shapiro(cohort_durations)
    print(f"\nShapiro-Wilk test for {cohort} durations:")
    print(f"Statistic: {stat:.3f}, p-value: {p_value:.3f}")

# Test selection based on normality
diagnostic_stat, diagnostic_p = stats.shapiro(diagnostic_durations)
control_stat, control_p = stats.shapiro(control_durations)
both_normal = diagnostic_p > 0.05 and control_p > 0.05

print("\nTest selection for durations:")
print(f"diagnostic-images normality: p = {diagnostic_p:.3f} ({'normal' if diagnostic_p > 0.05 else 'non-normal'})")
print(f"cross-section normality: p = {control_p:.3f} ({'normal' if control_p > 0.05 else 'non-normal'})")
print(f"Using {'t-test' if both_normal else 'Mann-Whitney U test'} based on normality results")

if both_normal:
    # Use t-test for normal distributions
    stat, p_value = stats.ttest_ind(control_durations, diagnostic_durations)
    print("\nBoth groups are normally distributed, using Independent t-test:")
    print(f"t-statistic: {stat:.3f}, p-value: {p_value:.3f}")
    
    # Calculate Cohen's d effect size for t-test
    n1, n2 = len(control_durations), len(diagnostic_durations)
    pooled_std = np.sqrt(((n1 - 1) * control_durations.std()**2 + (n2 - 1) * diagnostic_durations.std()**2) / (n1 + n2 - 2))
    effect_size = abs(control_durations.mean() - diagnostic_durations.mean()) / pooled_std
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
else:
    # Use Mann-Whitney U test for non-normal distributions
    stat, p_value = stats.mannwhitneyu(control_durations, diagnostic_durations, alternative='two-sided')
    print("\nOne or both groups are non-normal, using Mann-Whitney U test:")
    print(f"U-statistic: {stat:.3f}, p-value: {p_value:.3f}")
    
    # Calculate effect size r = Z/√N for Mann-Whitney U test
    n1, n2 = len(control_durations), len(diagnostic_durations)
    z_score = stats.norm.ppf(p_value/2)
    effect_size = abs(z_score / np.sqrt(n1 + n2))
    print(f"Effect size (r): {effect_size:.3f}")

# 4.4 Question Group Analysis
print("\nQuestion Group Analysis:")

# Prepare question group data
question_groups_dict = question_groups.set_index('Question')['QuestionGroup'].to_dict()

# Calculate number of tests for Bonferroni correction
n_tests = len(set(question_groups_dict.values()))
bonferroni_alpha = 0.05 / n_tests
print(f"\nBonferroni-corrected significance level: {bonferroni_alpha:.4f} (0.05/{n_tests})")

# Calculate scores and effect sizes by question group
group_results = []
for group in set(question_groups_dict.values()):
    questions = [q for q, g in question_groups_dict.items() if g == group]
    group_scores = test_results_clean[questions].mean(axis=1)
    
    control_group = group_scores[test_results_clean['Cohort'] == 'cross-section']
    diagnostic_group = group_scores[test_results_clean['Cohort'] == 'diagnostic-images']
    
    # Calculate effect size and CI
    try:
        stat, p_value = stats.mannwhitneyu(control_group, diagnostic_group, alternative='two-sided')
        n1, n2 = len(control_group), len(diagnostic_group)
        
        if n1 > 0 and n2 > 0:
            z_score = stats.norm.ppf(p_value/2)
            effect_size = abs(z_score / np.sqrt(n1 + n2))
            
            # Calculate 95% CI for effect size
            se = np.sqrt((n1 + n2) / (n1 * n2))
            ci_lower = effect_size - 1.96 * se
            ci_upper = effect_size + 1.96 * se
        else:
            effect_size = ci_lower = ci_upper = float('nan')
            p_value = float('nan')
    except Exception as e:
        print(f"\nCould not calculate effect size for group {group}: {str(e)}")
        effect_size = ci_lower = ci_upper = float('nan')
        p_value = float('nan')
    
    group_results.append({
        'Group': group,
        'Effect Size': effect_size,
        'CI Lower': ci_lower,
        'CI Upper': ci_upper,
        'p-value': p_value
    })

# Create forest plot
plt.figure(figsize=(12, 6))
y_pos = np.arange(len(group_results))

# Plot effect sizes and CIs
for i, result in enumerate(group_results):
    plt.plot([result['CI Lower'], result['CI Upper']], [i, i], 'b-')
    plt.plot(result['Effect Size'], i, 'bs')

# Customize plot
plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
plt.yticks(y_pos, [r['Group'] for r in group_results])
plt.xlabel('Effect Size (r)')
plt.title('Forest Plot of Effect Sizes by Question Group')

# Add annotations for p-values
for i, result in enumerate(group_results):
    plt.text(plt.xlim()[1], i, f"p = {result['p-value']:.3f}", 
             verticalalignment='center')

plt.tight_layout()
plt.savefig('figures/forest_plot.png')
plt.close()

# 5. Post-test Survey Analysis
print("\n=== Step 5: Post-test Survey Analysis ===\n")

# Likert Scale Analysis
print("Likert Scale Analysis:")
for q in ['Q4_1', 'Q4_2', 'Q4_3']:
    control_responses = posttest_survey_clean[posttest_survey_clean['Cohort'] == 'Control'][f'{q}_numeric']
    experimental_responses = posttest_survey_clean[posttest_survey_clean['Cohort'] == 'Experimental'][f'{q}_numeric']
    
    stat, p_value = stats.mannwhitneyu(control_responses, experimental_responses, alternative='two-sided')
    n1, n2 = len(control_responses), len(experimental_responses)
    z_score = stats.norm.ppf(p_value/2)
    effect_size = abs(z_score / np.sqrt(n1 + n2))
    
    print(f"\n{q}:")
    print(f"Mann-Whitney U test: U = {stat:.3f}, p = {p_value:.3f}")
    print(f"Effect size (r): {effect_size:.3f}")

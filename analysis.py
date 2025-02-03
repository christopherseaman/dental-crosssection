import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestPower

# Utility Functions
def duration_to_seconds(duration_str):
    """Convert duration string (e.g., '4m 59s 0ms') to seconds."""
    if 'm' not in duration_str:
        return None
    parts = duration_str.split('m')
    minutes = int(parts[0])
    seconds = 0
    if 's' in parts[1]:
        seconds = int(parts[1].split('s')[0].strip())
    return minutes * 60 + seconds

def test_normality(data, group_col=None):
    """Test normality of data, optionally by group."""
    if group_col is None:
        stat, p_value = stats.shapiro(data)
        return p_value > 0.05
    
    results = {}
    for group in data[group_col].unique():
        group_data = data[data[group_col] == group]
        stat, p_value = stats.shapiro(group_data)
        results[group] = {'p_value': p_value, 'normal': p_value > 0.05}
        print(f"Shapiro-Wilk test for {group}:")
        print(f"Statistic: {stat:.3f}, p-value: {p_value:.3f}")
    return results

def compare_groups(group1, group2, test_type='auto', alternative='two-sided'):
    """Compare two groups using appropriate statistical test."""
    if test_type == 'auto':
        # Test normality
        normal1 = stats.shapiro(group1)[1] > 0.05
        normal2 = stats.shapiro(group2)[1] > 0.05
        test_type = 't-test' if normal1 and normal2 else 'mann-whitney'
    
    if test_type == 't-test':
        stat, p_value = stats.ttest_ind(group1, group2)
        effect_size = calculate_cohens_d(group1, group2)
        test_name = "Independent t-test"
    else:
        stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        effect_size = calculate_mann_whitney_r(group1, group2, p_value)
        test_name = "Mann-Whitney U test"
    
    print(f"\n{test_name}:")
    print(f"Statistic: {stat:.3f}, p-value: {p_value:.3f}")
    print(f"Effect size: {effect_size:.3f}")
    
    return stat, p_value, effect_size

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / 
                        (n1 + n2 - 2))
    return abs(group1.mean() - group2.mean()) / pooled_std

def calculate_mann_whitney_r(group1, group2, p_value):
    """Calculate effect size r for Mann-Whitney U test."""
    n1, n2 = len(group1), len(group2)
    z_score = stats.norm.ppf(p_value/2)
    return abs(z_score / np.sqrt(n1 + n2))

# Visualization Functions
def setup_plotting():
    """Set up plotting style and create figures directory."""
    sns.set_theme(style="whitegrid")
    if not os.path.exists('figures'):
        os.makedirs('figures')

def plot_distribution_by_cohort(data, value_col, cohort_col, title, xlabel, filename):
    """Create and save distribution plot comparing cohorts."""
    plt.figure(figsize=(10, 6))
    for cohort in data[cohort_col].unique():
        values = data[data[cohort_col] == cohort][value_col]
        sns.kdeplot(data=values, label=cohort, fill=True, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'figures/{filename}.png')
    plt.close()

def create_summary_table(data, value_col, cohort_col=None):
    """Create summary statistics table overall and optionally by cohort."""
    if cohort_col:
        stats_by_group = []
        for group in data[cohort_col].unique():
            group_data = data[data[cohort_col] == group][value_col]
            stats = {
                'Group': group,
                'N': len(group_data),
                'Mean': group_data.mean(),
                'SD': group_data.std(),
                'Min': group_data.min(),
                'Q1': group_data.quantile(0.25),
                'Median': group_data.median(),
                'Q3': group_data.quantile(0.75),
                'Max': group_data.max(),
                'IQR': group_data.quantile(0.75) - group_data.quantile(0.25)
            }
            stats_by_group.append(stats)
        return pd.DataFrame(stats_by_group)
    else:
        stats = {
            'N': len(data),
            'Mean': data.mean(),
            'SD': data.std(),
            'Min': data.min(),
            'Q1': data.quantile(0.25),
            'Median': data.median(),
            'Q3': data.quantile(0.75),
            'Max': data.max(),
            'IQR': data.quantile(0.75) - data.quantile(0.25)
        }
        return pd.DataFrame([stats])

def create_likert_counts_table(data, question):
    """Create table of response counts for a single Likert question."""
    categories = ['Strongly disagree', 'Somewhat disagree', 
                 'Neither agree nor disagree', 
                 'Somewhat agree', 'Strongly agree']
    
    # Create counts for each cohort
    cohorts = ['cross-section', 'diagnostic-images']
    counts_by_cohort = []
    
    for cohort in cohorts:
        cohort_data = data[data['Cohort'] == cohort]
        counts = cohort_data[question].value_counts()
        # Ensure all categories are present
        counts = counts.reindex(categories, fill_value=0)
        counts.name = cohort
        counts_by_cohort.append(counts)
    
    # Combine into a DataFrame
    return pd.DataFrame(counts_by_cohort, index=cohorts)

def plot_score_duration_correlation(data, filename):
    """Create scatter plot of score vs duration with regression line."""
    plt.figure(figsize=(10, 6))
    sns.regplot(x='DurationMinutes', y='ScoreNumeric', data=data,
                scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    plt.title('Score vs Duration')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Score')
    plt.savefig(f'figures/{filename}.png')
    plt.close()

def plot_boxplot(data, x_col, y_col, title, xlabel, ylabel, filename):
    """Create and save boxplot."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_col, y=y_col, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'figures/{filename}.png')
    plt.close()

def create_single_likert_plot(data, question, question_label, title, filename, by_cohort=False):
    """Create and save a Likert plot for a single question."""
    categories = ['Strongly disagree', 'Somewhat disagree', 
                 'Neither agree nor disagree', 
                 'Somewhat agree', 'Strongly agree']
    
    if by_cohort:
        # Create separate series for each cohort
        cohorts = ['cross-section', 'diagnostic-images']
        results = []
        for cohort in cohorts:
            cohort_data = data[data['Cohort'] == cohort]
            d = cohort_data[question].value_counts(normalize=True) * 100
            d = d.reindex(categories).fillna(0)
            results.append(d)
        df = pd.DataFrame(results, index=cohorts)
    else:
        # Create single series for all data
        d = data[question].value_counts(normalize=True) * 100
        d = d.reindex(categories).fillna(0)
        df = pd.DataFrame([d], index=['All Students'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['#ca0020', '#f4a582', '#f7f7f7', '#92c5de', '#0571b0']
    neutral_idx = len(categories) // 2
    
    # Plot bars for each row (cohort or overall)
    for row_idx, row_label in enumerate(df.index):
        left = 0
        for cat_idx, category in enumerate(categories):
            width = df.loc[row_label, category]
            ax.barh(row_idx, width, left=left, color=colors[cat_idx],
                   label=category if row_idx == 0 else "")
            # Add percentage label if >= 5%
            if width >= 5:
                ax.text(left + width/2, row_idx, f'{width:.0f}%',
                       ha='center', va='center')
            left += width
    
    # Customize plot
    ax.set_title(f"{title}\n{question_label}", pad=20)
    ax.set_xlabel('Percentage of Responses')
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    
    # Add legend at the bottom
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=len(categories),
             title='Response Scale')
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()

# Main Analysis
def main():
    # Define constants
    likert_questions = ['Q4_1', 'Q4_2', 'Q4_3']
    question_labels = [
        'Prepared for interpreting\nradiological images',
        'Satisfied with presentation\nof anatomical structures',
        'More confident in identifying\nstructures on images'
    ]
    
    # Data Import
    print("\n=== Step 1: Data Import ===\n")
    test_results = pd.read_csv('data/test_results.tsv', sep='\t')
    question_groups = pd.read_csv('data/question_groups.tsv', sep='\t', 
                                skiprows=1, names=['Question', 'QuestionGroup'])
    posttest_survey = pd.read_csv('data/posttest_survey.tsv', sep='\t')
    
    # Data Preprocessing
    print("\n=== Step 2: Data Quality & Preprocessing ===\n")
    
    # Convert durations and scores
    test_results['DurationSeconds'] = test_results['TestDuration'].apply(duration_to_seconds)
    test_results['ScoreNumeric'] = test_results['Score'].str.rstrip('%').astype(float) / 100
    test_results['DurationMinutes'] = test_results['DurationSeconds'] / 60
    
    # Convert Likert scales
    likert_mapping = {
        'Strongly disagree': 1, 'Somewhat disagree': 2,
        'Neither agree nor disagree': 3, 'Somewhat agree': 4,
        'Strongly agree': 5
    }
    for col in ['Q4_1', 'Q4_2', 'Q4_3']:
        posttest_survey[f'{col}_numeric'] = posttest_survey[col].map(likert_mapping)
    
    # Clean data
    test_results_clean = test_results.copy()
    posttest_survey_clean = posttest_survey.dropna(subset=['Cohort'])
    
    # Print data quality info
    print("\nCohort sizes in test_results_clean:")
    print(test_results_clean['Cohort'].value_counts())
    print("\nCohort sizes in posttest_survey_clean:")
    print(posttest_survey_clean['Cohort'].value_counts())
    
    # Setup plotting
    setup_plotting()
    
    # Score Analysis
    print("\n=== Step 3: Score Analysis ===\n")
    control_scores = test_results_clean[test_results_clean['Cohort'] == 'cross-section']['ScoreNumeric']
    diagnostic_scores = test_results_clean[test_results_clean['Cohort'] == 'diagnostic-images']['ScoreNumeric']
    
    # Compare scores
    score_stat, score_p, score_effect = compare_groups(control_scores, diagnostic_scores)
    
    # Score visualizations
    plot_distribution_by_cohort(test_results_clean, 'ScoreNumeric', 'Cohort',
                              'Score Distribution by Cohort', 'Score (%)',
                              'score_distribution')
    plot_boxplot(test_results_clean, 'Cohort', 'ScoreNumeric',
                'Score Distribution Box Plot by Cohort', 'Cohort', 'Score (%)',
                'score_boxplot')
    
    # Duration Analysis
    print("\n=== Step 4: Duration Analysis ===\n")
    control_durations = test_results_clean[test_results_clean['Cohort'] == 'cross-section']['DurationSeconds']
    diagnostic_durations = test_results_clean[test_results_clean['Cohort'] == 'diagnostic-images']['DurationSeconds']
    
    # Compare durations
    dur_stat, dur_p, dur_effect = compare_groups(control_durations, diagnostic_durations)
    
    # Duration visualizations
    plot_distribution_by_cohort(test_results_clean, 'DurationMinutes', 'Cohort',
                              'Test Duration Distribution by Cohort', 'Duration (minutes)',
                              'duration_distribution')
    plot_boxplot(test_results_clean, 'Cohort', 'DurationMinutes',
                'Test Duration Distribution Box Plot by Cohort', 'Cohort', 'Duration (minutes)',
                'duration_boxplot')
    
    # Question Group Analysis
    print("\n=== Step 5: Question Group Analysis ===\n")
    
    # Prepare question group data
    question_groups_dict = question_groups.set_index('Question')['QuestionGroup'].to_dict()
    
    # Calculate number of tests for Bonferroni correction
    n_tests = len(set(question_groups_dict.values()))
    bonferroni_alpha = 0.05 / n_tests
    print(f"Bonferroni-corrected significance level: {bonferroni_alpha:.4f} (0.05/{n_tests})")
    
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
    
    # Print question group analysis results
    print("\nQuestion Group Analysis:")
    for result in group_results:
        group = result['Group']
        effect = result['Effect Size']
        p_val = result['p-value']
        sig = "significant" if p_val < bonferroni_alpha else "not significant"
        print(f"\n{group}:")
        print(f"Effect size: {effect:.3f}")
        print(f"p-value: {p_val:.3f} ({sig} at α = {bonferroni_alpha:.3f})")
    
    # Summary Tables
    print("\n=== Step 6: Summary Tables ===\n")
    
    print("Score Summary Statistics:")
    score_stats = create_summary_table(test_results_clean, 'ScoreNumeric', 'Cohort')
    print(score_stats.round(3))
    
    print("\nDuration Summary Statistics (minutes):")
    duration_stats = create_summary_table(test_results_clean, 'DurationMinutes', 'Cohort')
    print(duration_stats.round(3))
    
    # Print Likert response counts
    print("\nLikert Response Counts by Question:")
    for q, label in zip(likert_questions, question_labels):
        print(f"\n{q} - {label}:")
        likert_counts = create_likert_counts_table(posttest_survey_clean, q)
        print(likert_counts)
    
    # Create score vs duration correlation plot
    plot_score_duration_correlation(test_results_clean, 'score_vs_duration')
    
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
    
    # Correlation Analysis
    print("\n=== Step 7: Correlation Analysis ===\n")
    score_duration_normal = all(test_normality(test_results_clean[col]) 
                              for col in ['ScoreNumeric', 'DurationSeconds'])
    
    if score_duration_normal:
        correlation, p_value = stats.pearsonr(test_results_clean['ScoreNumeric'],
                                            test_results_clean['DurationSeconds'])
        print("Using Pearson correlation (normal data)")
    else:
        correlation, p_value = stats.spearmanr(test_results_clean['ScoreNumeric'],
                                             test_results_clean['DurationSeconds'])
        print("Using Spearman correlation (non-normal data)")
    
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"P-value: {p_value:.3f}")
    
    # Likert Analysis
    print("\n=== Step 8: Likert Analysis ===\n")
    
    # Statistical analysis of Likert responses
    likert_results = []
    
    # Create individual plots for each question
    for q, label in zip(likert_questions, question_labels):
        # Overall plot for this question
        create_single_likert_plot(
            posttest_survey_clean, q, label,
            'Student Feedback - All Groups',
            f'likert_q{q[-1]}_overall'
        )
        
        # Cohort comparison plot for this question
        create_single_likert_plot(
            posttest_survey_clean, q, label,
            'Student Feedback by Cohort',
            f'likert_q{q[-1]}_by_cohort',
            by_cohort=True
        )
        
        # Statistical comparison
        control_responses = posttest_survey_clean[posttest_survey_clean['Cohort'] == 'cross-section'][f'{q}_numeric']
        experimental_responses = posttest_survey_clean[posttest_survey_clean['Cohort'] == 'diagnostic-images'][f'{q}_numeric']
        
        stat, p_value = stats.mannwhitneyu(control_responses, experimental_responses, alternative='two-sided')
        n1, n2 = len(control_responses), len(experimental_responses)
        z_score = stats.norm.ppf(p_value/2)
        effect_size = abs(z_score / np.sqrt(n1 + n2))
        
        print(f"\n{q}:")
        print(f"Mann-Whitney U test: U = {stat:.3f}, p = {p_value:.3f}")
        print(f"Effect size (r): {effect_size:.3f}")
        
        likert_results.append((q, p_value, effect_size))
    
    print("\nAnalysis complete. All visualizations have been saved to the 'figures' directory.")

if __name__ == "__main__":
    main()

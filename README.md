# Cross-Sectional Anatomy Curriculum Project

## Project Description and Abstract

This project intends to develop an integrated cross-sectional anatomy curriculum within the first-year anatomy course at UCSF to prime dental students with spatial ability skills and improve diagnostic imaging interpretation capabilities. Annotations on large cross-sectional foam boards will allow learners to collaborate as they understand anatomical structures in different dimensions and planes, leveraging the advantages of visuospatial modalities that cross-sections provide to bridge the gap between anatomical knowledge and clinical practice. Assessments will determine the intervention's strengths regarding diagnostic interpretation capabilities and surveys will underscore self-reported confidence in preparation for clinical exercises.

## Methods

A needs assessment survey will be offered to dental students to understand their level of preparedness for interpreting diagnostic images following the anatomy and radiology courses they took. During the lab, first-year dental students annotate unlabeled posterboards, working with their lab groups. The students then rotate around the other group's posterboards to fill in or correct labels before answers are revealed. Participants are randomly split into 2 groups: one experimental group that will be asked to identify structures on radiological images only and one control group that will take a similar test to the pretest. Lastly, a posttest survey is intended at gauging student satisfaction and feedback regarding the cross-sectional curriculum, as well as the experimental test's ability in measuring knowledge acquisition.

## Analysis Plan

1. Import data
   1. Test results, `data/question_groups.tsv`
   2. Question grouping, `data/question_groups.tsv`
   3. Post-test survey, `data/posttest_survey.tsv`
      1. Q2: "Choose all that apply..."
      2. Q4_1: "This cross-section lab period prepared me for interpreting anatomical structures on radiological images."
      3. Q4_2: "I am satisfied with the current presentation of anatomical structures in this lab period."
      4. Q4_3: "I feel more confident in my abilities to identify anatomical structures on radiological images after this lab period."
      5. Q7: "What changes, if any, would you like to see in this cross-section lab to help you prepare for interpreting radiological images?"
      6. Q6: "Please feel free to share your experience with this cross-section lab below"

2. Data Quality & Preprocessing
   1. Check for missing values, duplicates, data types
   2. Test result TestDuration -> time in seconds
   3. Post-test survey Q4_1, Q4_2, Q4_3 Likert scales -> 1-5
   4. Validation
      1. Question groups map to questions
      2. Durations of reasonable length (0 < x < 3600 seconds)
      3. Scores calculated correctly (question columns recorded 1/0 for correct/incorrect)
   5. Outlier detection for test durations (boxplot method)

3. Descriptive Statistics & Visualizations
   1. Results: scores and durations
      1. Table (mean, sd, range, IQR) overall and split by cohort
      2. Histogram by cohort overlaid on the same graph
      3. Histogram by cohort, 2 graphs side-by-side with boxplot overlaid
      4. Scatter plot: duration vs. score with regression line
   2. Posttest: Q4_1, Q4_2, Q4_3
      1. Likert histogram comparing cohorts
      2. Diverging bar chart for Likert responses

4. Statistical Analysis
   1. Test Score Analysis
      1. Normality test (Shapiro-Wilk) for each cohort
      2. Based on normality:
         - If normal: Independent t-test
         - If non-normal: Mann-Whitney U test
      3. Effect size calculation:
         - For t-test: Cohen's d
         - For Mann-Whitney: r = Z/√N
      4. 95% confidence intervals for mean/median difference

   2. Duration Analysis
      1. Normality test (Shapiro-Wilk)
      2. Choose test based on results (t-test or Mann-Whitney)
      3. Effect size and confidence intervals

   3. Correlation Analysis
      1. Test score vs. duration
         1. Normality test for both variables
         2. Choose correlation test:
            - If normal: Pearson
            - If non-normal: Spearman
         3. Calculate correlation coefficient and p-value
         4. Plot correlation with confidence bands

   4. Question Group Analysis
      1. Normality tests per group
      2. Multiple comparison correction (Bonferroni)
      3. Effect sizes for each comparison
      4. Visualization: Forest plot of effect sizes

5. Post-test Survey Analysis
   1. Likert Scale Questions (Q4_1, Q4_2, Q4_3)
      1. Convert responses to numeric scale (1-5)
         - Strongly disagree = 1
         - Somewhat disagree = 2
         - Neither agree nor disagree = 3
         - Somewhat agree = 4
         - Strongly agree = 5
      2. Statistical Analysis
         - Mann-Whitney U tests (appropriate for ordinal Likert data)
         - Effect sizes (r = Z/√N) for meaningful differences
         - Bonferroni correction for multiple comparisons
      3. Visualization
         - Diverging stacked bar charts
         - Negative responses (disagree) extend left from center
         - Positive responses (agree) extend right from center
         - Neutral responses centered
         - Percentage labels for responses ≥ 5%
         - Separate plots by cohort and combined
      4. Interpretation
         - Compare response distributions between cohorts
         - Consider both statistical significance and effect sizes
         - Account for multiple comparisons in significance testing
         - Evaluate practical significance of differences

6. Brief summary of methods and results in README.md

## Heads

### `posttest_survey.tsv` 

```
StartDate	EndDate	ID	Cohort	Q2	Q4_1	Q4_2	Q4_3	Q7	Q6
12/3/24 11:50	12/3/24 11:51	S2841	Control		Somewhat agree	Somewhat agree	Strongly agree	I think it would be helpful to practice differentiating different types of images MRI vs CT. 	I had a great time! Thanks Daania! 
```

### `test_results.tsv` 

```
StartDate	EndDate	TestDuration	ID	Cohort	Q3	Q4	Q16	Q17	Q19	Q20	Q22	Q23	Q25	Q26	Q28	Q29	Q31	Q32	Q34	Q35	NumberCorrect	Score
12/3/24 11:44	12/3/24 11:49	4m 59s 0ms	S2852	diagnostic-images	1	0	1	0	0	1	1	1	1	1	1	1	1	1	1	1	13	81%
```

### `question_groups.tsv` 

```
Question  QuestionGroup
Q3	Axial
Q4	Axial
```

## Results

### Key Findings

1. Test Performance
   - Cross-section group (n=32) significantly outperformed diagnostic-images group (n=31)
   - Mean scores: 72.2% ± 14.8% vs 54.5% ± 15.4%
   - Large effect size (Cohen's d = 1.170, p < 0.001)

2. Test Duration
   - No significant difference between groups
   - Cross-section: 7.8 ± 2.2 minutes
   - Diagnostic-images: 8.5 ± 2.5 minutes
   - Small effect size (r = 0.153, p = 0.167)

3. Score-Duration Relationship
   - Weak negative correlation (ρ = -0.185)
   - Not statistically significant (p = 0.167)

4. Student Feedback (Likert Scale)
   - Preparation for radiological interpretation (Q4_1): No difference between groups
   - Satisfaction with presentation (Q4_2): Significant difference favoring control group (p = 0.016, r = 0.313)
   - Confidence in structure identification (Q4_3): No significant difference (p = 0.167, r = 0.180)

### Generated Visualizations

1. Score Distribution (`figures/score_distribution.png`)
   - Kernel density plots comparing score distributions between cohorts
   - Shows clear separation between groups

2. Duration Distribution (`figures/duration_distribution.png`)
   - Kernel density plots comparing test duration distributions
   - Shows substantial overlap between groups

3. Score Boxplots (`figures/score_boxplot.png`)
   - Box-and-whisker plots comparing score distributions
   - Highlights median, quartiles, and potential outliers

4. Duration Boxplots (`figures/duration_boxplot.png`)
   - Box-and-whisker plots comparing test duration distributions
   - Identifies duration outliers

5. Score vs Duration (`figures/score_vs_duration.png`)
   - Scatter plot with regression line and 95% confidence bands
   - Visualizes weak negative correlation

6. Forest Plot (`figures/forest_plot.png`)
   - Effect sizes and confidence intervals for question groups
   - Compares performance across different anatomical planes

7. Likert Responses
   - By cohort (`figures/likert_responses_cross-section.png`, `figures/likert_responses_diagnostic-images.png`)
   - Combined (`figures/likert_responses_combined.png`)
   - Diverging stacked bar charts that display responses on a horizontal axis, with negative responses extending left from center and positive responses extending right
   - The Likert analysis revealed that students in both cohorts generally responded positively to the cross-section lab. Statistical comparison using Mann-Whitney U tests showed no significant differences between cohorts for preparedness (Q4_1) or confidence (Q4_3), but found that the cross-section group reported higher satisfaction with the presentation of anatomical structures (Q4_2, p = 0.016, r = 0.313). This suggests that while both teaching methods were effective for building confidence and preparedness, students preferred the traditional cross-sectional approach for structure presentation.
   - Note: Q4_1 was answered "Somewhat Agree" by all respondents

## Analysis Output

python analysis.py

=== Step 1: Data Import ===


=== Step 2: Data Quality & Preprocessing ===


Cohort sizes in test_results_clean:
Cohort
cross-section        32
diagnostic-images    31
Name: count, dtype: int64

Cohort sizes in posttest_survey_clean:
Cohort
diagnostic-images    30
cross-section        29
Name: count, dtype: int64

=== Step 3: Score Analysis ===


Independent t-test:
Statistic: 4.641, p-value: 0.000
Effect size: 1.170

=== Step 4: Duration Analysis ===


Mann-Whitney U test:
Statistic: 407.000, p-value: 0.224
Effect size: 0.153

=== Step 5: Question Group Analysis ===

Bonferroni-corrected significance level: 0.0167 (0.05/3)

=== Step 6: Correlation Analysis ===

Using Spearman correlation (non-normal data)
Correlation coefficient: -0.185
P-value: 0.146

=== Step 7: Likert Analysis ===


Q4_1:
Mann-Whitney U test: U = 435.000, p = 1.000
Effect size (r): 0.000

Q4_2:
Mann-Whitney U test: U = 581.500, p = 0.016
Effect size (r): 0.313

Q4_3:
Mann-Whitney U test: U = 518.500, p = 0.167
Effect size (r): 0.180

Analysis complete. All visualizations have been saved to the 'figures' directory.
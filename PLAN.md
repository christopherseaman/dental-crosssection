# Cross-Sectional Anatomy Curriculum Project

## Project Description and Abstract

This project intends to develop an integrated cross-sectional anatomy curriculum within the first-year anatomy course at UCSF to prime dental students with spatial ability skills and improve diagnostic imaging interpretation capabilities. Annotations on large cross-sectional foam boards will allow learners to collaborate as they understand anatomical structures in different dimensions and planes, leveraging the advantages of visuospatial modalities that cross-sections provide to bridge the gap between anatomical knowledge and clinical practice. Assessments will determine the intervention's strengths regarding diagnostic interpretation capabilities and surveys will underscore self-reported confidence in preparation for clinical exercises.

## Methods

A needs assessment survey will be offered to dental students to understand their level of preparedness for interpreting diagnostic images following the anatomy and radiology courses they took. During the lab, first-year dental students annotate unlabeled posterboards, working with their lab groups. The students then rotate around the other group's posterboards to fill in or correct labels before answers are revealed. Participants are randomly split into 2 groups: one experimental group that will be asked to identify structures on radiological images only and one control group that will take a similar test to the pretest. Lastly, a posttest survey is intended at gauging student satisfaction and feedback regarding the cross-sectional curriculum, as well as the experimental test's ability in measuring knowledge acquisition.

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
2. Transform
   1. Test result TestDuration -> time in seconds
   2. Post-test survey Q4_1, Q4_2, Q4_3 Likert scales -> 1-5
3. Tables & Graphs
   1. Results: scores and durations, table (mean, sd, range, IQR) overall and split by cohort, graph (histogram by cohort)
   2. Posttest: Q4_1, Q4_2, Q4_3; Likert histogram comparing cohorts
4. Cohort Comparison
   1. Results
      1. Overall
         1. Score, student's t-test
         2. Duration, Mann-Whitney
      2. By question group
         1. Score, student's t-test
         2. Duration, Mann-Whitney
   2. Post-test Survey
      1. Likert's Q4_1, Q4_2, Q4_3, 
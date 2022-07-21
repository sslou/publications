# Temporal Associations Between EHR-Derived Workload, Burnout, and Errors: a Prospective Cohort Study

Code to reproduce audit log-derived workload measurements and statistical models described in [Lou et al., *J. Gen. Intern. Med.*, 2022](https://doi.org/10.1007/s11606-022-07620-3).

## Files
- measure_access_logs.py
    - input: raw access log files
    - measures various characteristics of participants' EHR use in the `window` # of days preceding each survey completion, including total EHR time, after-hours EHR time, time spent on notes, chart review etc.
- variable_definitions.txt
    - documentation for the output variables measured by `measure_access_logs.py`
- metric_categorized.csv
    - manually categorized METRIC_NAME, REPORT_NAME pairs, used primarily to summarize time spent on data review as the sum of time spent on "Note Review", "Chart Review", and "Results Review".
    - also shows the total number of occurrences of each action (METRIC_NAME, REPORT_NAME pair) across the entire study, and the aggregate total number of hours spent by all participants on each action.
- burnout_lmm_modeling.Rmd
    - R script demonstrating how to fit the linear mixed effect model and poison models described in the manuscript. note that the models described in the manuscript were fit with SAS. this example code is provided just to demonstrate the model formulation.

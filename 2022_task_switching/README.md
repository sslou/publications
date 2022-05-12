# Effect of Attention Switching on Clinician Workload and Wrong-Patient Errors

Code to reproduce EHR audit log-derived attention switching metric and statistical analysis from [Lou et al., *Br J Anaesth.*, 2022](https://doi.org/10.1016/j.bja.2022.04.012).

## Files
- `measure_access_logs.py`
    - input: raw access log files (from `ACCESS_LOG` table in Epic Clarity)
    - measures various attention switching (patient switching) metrics, as well as total EHR time, and other EHR use measures aggregated at the level of individual clinician shifts in surgical intensive care units.
    - this is an extension of code documented more extensively in [2021_burnout_lmm](https://github.com/sslou/publications/tree/main/2021_burnout_lmm).
- `task_switching_statistics.Rmd`
    - R code to reproduce the statistical analysis described in the manuscript.

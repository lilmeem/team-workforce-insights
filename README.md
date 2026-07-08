# Team Workforce Insights — Career Compass

SIADS 699 capstone project. Job postings use inconsistent terminology to describe similar skills and qualifications, which makes it hard for job seekers to compare roles or see what's missing from their background. This project builds an NLP/ML pipeline that extracts and standardizes skills from job descriptions, recommends postings based on a user's skill profile, and highlights skill gaps relative to the market.

**Team:** Cory Vinlove, Tameem Syed

## Status

Early-stage exploratory data analysis. No cleaning, feature engineering, or modeling has happened yet.

- [`02_eda_postings.ipynb`](02_eda_postings.ipynb) — EDA on the core `postings.csv` table: shape, dtypes, duplicates, missingness, categorical/numerical distributions.
- [`03_eda_companies_and_jobs.ipynb`](03_eda_companies_and_jobs.ipynb) — EDA on the supporting company/job/mapping tables, plus how well each one joins back to `postings.csv`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Then open either notebook in Jupyter or VS Code and run the cells top to bottom.

## Data access

This project uses the **LinkedIn Job Postings** dataset from Kaggle — likely [arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) based on file structure, but please confirm this matches what you downloaded before relying on it.

Download it from Kaggle and place the contents alongside the notebooks so the folder looks like:

```
.
├── postings.csv
├── companies/
│   ├── companies.csv
│   ├── company_industries.csv
│   ├── company_specialities.csv
│   └── employee_counts.csv
├── jobs/
│   ├── benefits.csv
│   ├── job_industries.csv
│   ├── job_skills.csv
│   └── salaries.csv
└── mappings/
    ├── industries.csv
    └── skills.csv
```

The raw data is not committed to this repository (`postings.csv` alone is ~500MB, and licensing terms are still being confirmed — see open questions below). Check the dataset's Kaggle listing for its current license before redistributing any derived data.

## Open questions (tracked, not yet resolved)

- Whether `remote_allowed` missingness means "not remote" or "unspecified"
- How to handle salary outliers and reconcile `postings.csv`'s salary columns against the more-complete `jobs/salaries.csv`
- Whether placeholder values (e.g. `country == "0"` in `companies.csv`) need explicit handling beyond standard null checks
- What the `inferred` flag in `jobs/benefits.csv` actually represents
- Confirming the dataset's redistribution license before any derived data is shared publicly

## Roadmap

1. Resolve the open questions above and clean `postings.csv`
2. Build a skill-extraction pipeline over the `description` text (structured skill fields only cover 35 broad categories, not real skills)
3. Recommendation engine: match user skills to postings, surface skill gaps
4. Dashboard / interface
5. Evaluation (quantitative + qualitative)
6. Final report, video, and repo cleanup per the capstone spec checklist

# AIRA Deployment — Exact Steps

## GitHub structure
```text
AI-Academic-Advisor_2.0/
├── app.py
├── requirements.txt
├── README.md
├── DEPLOYMENT.md
├── evaluation_cases.csv
├── data/
│   ├── CLEANED_ACADEMIC_DATABASE.xlsx
│   ├── course_aliases.csv
│   ├── policy_documents.csv
│   ├── Student_Handbook_August_2026.pdf
│   ├── SOP_STUDENT_17082026_FINAL.pdf
│   └── source_originals/
│       ├── Minor_Courses_for_BTech_Students(3).xlsx
│       └── Semester_Spread_Structures_Sept_2026(3).xlsx
└── documentation / report / presentation files
```

## Why the cleaned workbook is used
Do **not** make the app retrieve directly from the raw Excel layouts. The cleaned workbook is the runtime structured database. The raw Excel files are kept under `source_originals/` only so the submission preserves provenance and can be audited back to the supplied sources.

## Streamlit Community Cloud
1. Push the complete project to GitHub.
2. Open Streamlit Community Cloud.
3. Select **Create app / Deploy**.
4. Repository: `2023vusuresh/AI-Academic-Advisor_2.0`.
5. Branch: `main`.
6. Main file: `app.py`.
7. Deploy.
8. Open **App settings → Secrets**.
9. Add:

```toml
GEMINI_API_KEY = "YOUR_NEW_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
```

10. Save secrets and restart/redeploy the app.

## First deployment test
Run these in order:

1. `What is the prerequisite for Financial and Management Accounting?`
2. `What is the prerequisite for Communication Skills?`
3. `I want to take Internet of Things next semester, what should I complete first?`
4. `Which courses depend on Data Structures?`
5. `What is the attendance requirement?`

Expected behaviour:
- FAMA → course found; prerequisite not specified.
- Communication Skills → asks whether UCOR103 or UCOR107.
- Internet of Things → lists the verified prerequisite courses.
- Data Structures → lists dependent courses.
- Attendance → retrieves Handbook/SOP evidence and shows PDF page sources.

## API-key safety
Never place a Gemini API key in `app.py`, `.env`, GitHub, README, screenshots, or the ZIP. If an old key was exposed anywhere, revoke it and create a new one before deployment.

# AIRA — AI Academic Advisor | Final Submission

A production-oriented academic decision-support prototype that knows when to answer, when to ask a specific follow-up question, and when the supplied evidence is insufficient.

## Runtime data architecture
The deployed application does **not** query the raw Excel layouts. The runtime knowledge layer is the cleaned, normalized database plus page-indexed PDF records:

```text
Supplied Excel + PDF sources
        ↓
Cleaning / normalization / audit
        ↓
CLEANED_ACADEMIC_DATABASE.xlsx + course_aliases.csv + policy_documents.csv
        ↓
Gemini query understanding
        ↓
LangGraph orchestration
        ↓
LangChain chunking + hybrid retrieval
        ↓
Deterministic source verification
        ↓
Specific clarification OR verified result
        ↓
Gemini grounded response + validation gate
```

## Runtime sources in `data/`
- `CLEANED_ACADEMIC_DATABASE.xlsx` — normalized academic tables derived from the supplied Excel workbooks.
- `Student_Handbook_August_2026.pdf` — supplied policy source.
- `SOP_STUDENT_17082026_FINAL.pdf` — supplied policy source.
- `policy_documents.csv` — page-indexed extraction of the supplied PDFs for retrieval.
- `course_aliases.csv` — canonical course-name/alias mapping.
- `synthetic_student_profiles.csv` and `student_course_history.csv` — synthetic/anonymized test data.

The original Excel workbooks are stored only under `source_archive/` for provenance and reproducibility. **The production application never reads those raw files.**

## Data-quality safeguards
- Unknown course-code placeholders such as `DON’T KNOW` are stored as an unknown code, never as a real course code.
- A missing/unknown prerequisite is never converted into `Nil`.
- `Nil`/`None` is treated as an explicit no-prerequisite value only when supported by the source.
- Duplicate course titles can trigger a specific course-selection follow-up.
- Batch/year differences are preserved rather than silently merged.
- Material conflicts are surfaced instead of arbitrarily choosing one record.
- PDF answers retain document and page provenance.

## Deployment
1. Push this repository to GitHub with the structure shown below.
2. In Streamlit Community Cloud select the repository, `main` branch, and `app.py`.
3. Open **Settings → Secrets** and add:

```toml
GEMINI_API_KEY = "YOUR_NEW_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
```

4. Save and let Streamlit rebuild the app.
5. Test the deployment with the validation queries below.

Never commit the Gemini API key to GitHub.

## Recommended deployment structure
```text
AI-Academic-Advisor_2.0/
├── app.py
├── requirements.txt
├── README.md
├── evaluation_cases.csv
├── data/
│   ├── CLEANED_ACADEMIC_DATABASE.xlsx
│   ├── Student_Handbook_August_2026.pdf
│   ├── SOP_STUDENT_17082026_FINAL.pdf
│   ├── policy_documents.csv
│   ├── course_aliases.csv
│   ├── synthetic_student_profiles.csv
│   └── student_course_history.csv
└── source_archive/
    ├── Minor_Courses_for_BTech_Students(3).xlsx
    └── Semester_Spread_Structures_Sept_2026(3).xlsx
```

## Validation queries
- `What is the prerequisite for Financial and Management Accounting?`
- `What is the prerequisite for Communication Skills?`
- `Which course is prerequisite for COMP302?`
- `Which courses depend on Data Structures?`
- `What is the attendance requirement?`
- `How long do I have to add or drop a course?`
- `What courses are in the Finance minor for 2022?`
- `Can SYN001 take UCOR205?`
- `Can I take COMP302?`

## Grounding principle
Gemini may understand the question and generate the final wording, but it is never allowed to supply an unverified academic fact. Structured verification is the source of truth. If verification cannot establish the answer, AIRA asks for the smallest useful clarification or states that the supplied sources are insufficient.

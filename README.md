# AIRA — AI Academic Advisor | Final Submission

## 1. Authoritative academic sources
AIRA uses exactly two academic source workbooks:

1. `data/Minor_Courses_for_BTech_Students(3).xlsx`
2. `data/Semester_Spread_Structures_Sept_2026(3).xlsx`

These are the only authoritative academic sources. Synthetic student profiles are test data only.

## 2. First stage: clean and organize the data
Before retrieval, the two Excel workbooks are processed into structured records:

- whitespace and text normalization
- course-code normalization
- semester-label normalization
- whitespace/non-breaking-space normalization while preserving raw source text
- course-code normalization for matching while preserving raw codes
- batch and semester organization from the workbook's own labels
- explicit distinction between `NIL`, blank/not-recorded, and `TBD` prerequisites
- preservation of `DON'T KNOW` and `NEW` course-code entries rather than invented codes
- separation of placeholder/TBD rows from active course records while retaining provenance and credits
- correct handling of the different structure-sheet layouts for 2022/2023 versus 2024/2025/2026
- exact-duplicate handling only; academic-year/batch versions are retained separately
- preservation of workbook, sheet and source-row metadata for evidence

The cleaned normalized tables are also provided in `data/cleaned/CLEANED_ACADEMIC_DATABASE.xlsx`. Source rows that contain a course title but no course code are preserved as title-only records; no code is invented. `DATA_CLEANING_AUDIT.csv` documents the cleaning rules and checks.

The source workbooks themselves are not modified; the cleaned records are a derived internal representation used for retrieval and verification.

## 3. Final architecture
`Excel sources → Clean & Organize → Structured records → Question routing/entity resolution → RAG retrieval → Hybrid reranking → Deterministic verification → Source-verified production answer + Sources`

Gemini is used in the required four-stage faculty/evaluation experiment. It is not an academic-data source and is never used as an ungrounded production fallback. Student-facing production answers come only from verified records derived from the two Excel workbooks.

## 4. Student-facing design
The normal interface is intentionally simple:

- ask in natural language
- course name is a first-class input; course code is optional
- optional synthetic student profile for eligibility questions
- quick visual question categories
- concise answer
- clear Sources section
- ambiguity, missing information and conflicts produce follow-up/uncertainty rather than guesses
- technical evaluation details are separated from the student view

## 5. Retrieval and coverage
AIRA supports source-grounded questions about:

- course names and codes
- prerequisites
- credits
- semester offerings/listings
- minor courses and L/T/P hours
- semester-wise course lists
- academic structure categories
- structure credit requirements
- course filters and lists supported by the workbooks
- synthetic-student prerequisite eligibility

If multiple course titles match, the system asks the user to choose. If multiple academic structures apply, it asks for the structure/year. If the supplied sources do not contain the requested fact, it says so.

## 6. RAG
Primary retrieval uses SentenceTransformer embeddings (`all-MiniLM-L6-v2`) with FAISS and hybrid reranking. A lexical TF-IDF fallback is explicitly labeled if dense dependencies are unavailable.

## 7. LLM
Gemini is accessed through the Google GenAI SDK. The key is read from Streamlit Secrets or the environment. Never hard-code the API key in the repository. Streamlit recommends storing secrets outside the Git repository and using the Community Cloud Secrets interface for deployed apps. 

## 8. Evaluation
The package contains `evaluation_cases.csv` with 30 predefined scenarios covering normal facts, course-name ambiguity, missing student context, eligibility, minors, semesters, structures, insufficient information, conflicts and unsupported questions.

The required four-stage comparison is:

`Basic LLM → Structured Prompting → LLM + RAG → LLM + RAG + Structured Student Data`

Do not claim 100% accuracy. Report measured benchmark results only when the benchmark has actually been executed and recorded.

## 9. Deployment
For Streamlit Community Cloud:

- repository root: `app.py`, `requirements.txt`, README, evaluation files
- Excel sources must be inside the `data/` folder
- main file: `app.py`
- branch: `main`
- add `GEMINI_API_KEY` through Streamlit Secrets

Do not upload API keys to GitHub.

## 10. Submission materials
- `AI_Academic_Advisor_FINAL_10_PAGE_REPORT.docx`
- `AI_Academic_Advisor_FINAL_10_PAGE_REPORT.pdf`
- `AI_Academic_Advisor_FINAL_PRESENTATION.pptx`
- `AI_Academic_Advisor_FINAL_SUBMISSION.ipynb`
- `app.py`
- `requirements.txt`
- `evaluation_cases.csv`
- `DATA_CLEANING_AUDIT.csv`
- `data/` with the two source workbooks

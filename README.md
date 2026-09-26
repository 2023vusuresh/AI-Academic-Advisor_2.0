# AIRA — AI Academic Advisor | DATA308 Assignment #1

A production-oriented academic decision-support prototype that knows when to answer, when to ask a specific follow-up question, and when the supplied information is insufficient.

## Authoritative source documents
The project has **four authoritative source documents**:

1. `data/source_originals/Minor_Courses_for_BTech_Students(3).xlsx`
2. `data/source_originals/Semester_Spread_Structures_Sept_2026(3).xlsx`
3. `data/Student_Handbook_August_2026.pdf`
4. `data/SOP_STUDENT_17082026_FINAL.pdf`

The two Excel workbooks are cleaned into `data/CLEANED_ACADEMIC_DATABASE.xlsx` before runtime retrieval. The Handbook and SOP are extracted into page-level `data/policy_documents.csv` while the original PDFs remain bundled for provenance.

### Important: raw vs cleaned data
**The app does not retrieve directly from the raw Excel layouts.**

- `CLEANED_ACADEMIC_DATABASE.xlsx` = runtime structured/canonical database.
- `course_aliases.csv` = canonical course-name/alias resolution.
- `policy_documents.csv` = page-level Handbook/SOP retrieval records.
- `source_originals/*.xlsx` and the two PDFs = original source evidence/provenance only.

This prevents inconsistent spreadsheet layouts and placeholder values from being sent directly to retrieval.

## Production architecture
`User → Gemini query understanding → LangGraph → entity resolution → LangChain chunking/RAG → hybrid retrieval → deterministic source verification → Gemini grounded response → validation gate → answer + evidence`

Specific course/prerequisite questions are verified against the canonical tables **before** a broad semantic executor. This prevents a valid title-only course such as Financial and Management Accounting (FAMA), whose source code is unknown, from being incorrectly reported as missing.

## Data-quality rules
- A course can exist even when its source course-code field is `DON’T KNOW`.
- `NIL/None` means an explicit no-prerequisite record.
- Blank/unknown prerequisite information means **not specified**, not “no prerequisite”.
- If multiple courses match the same title, AIRA asks which course the student means.
- If multiple academic structures/batches matter, AIRA asks for the applicable structure/year instead of guessing.
- Policy answers show PDF page-level evidence.

## Deployment
1. Upload the project files to GitHub with the `data/` folder intact.
2. Deploy `app.py` from Streamlit Community Cloud.
3. Add a newly generated Gemini API key to Streamlit Secrets:

```toml
GEMINI_API_KEY = "YOUR_NEW_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
```

Never commit the API key to GitHub. If a key has ever been pasted into a chat, screenshot, GitHub file, or log, revoke it and generate a new one.

## Demonstration queries
- `What is the prerequisite for Financial and Management Accounting?`
- `What is the prerequisite for FAMA?`
- `What is the prerequisite for Communication Skills?` → asks which Communication Skills course.
- `I want to take Internet of Things next semester, what should I complete first?`
- `Which courses depend on Data Structures?`
- `What is the attendance requirement?`
- `How long do I have to add or drop a course?`
- `Can I audit a course and still earn credits?`
- `What courses are in the Finance minor for 2022?`
- `Can SYN001 take UCOR205?`

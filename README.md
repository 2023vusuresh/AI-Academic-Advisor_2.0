# AIRA — AI Academic Advisor | DATA308 Assignment #1

A production-oriented academic decision-support prototype that knows when to answer, when to ask a follow-up question, and when the supplied information is insufficient.

## Included authoritative sources
- `data/Minor_Courses_for_BTech_Students(3).xlsx`
- `data/Semester_Spread_Structures_Sept_2026(3).xlsx`
- `data/Student_Handbook_August_2026.pdf`
- `data/SOP_STUDENT_17082026_FINAL.pdf`

The Handbook and SOP are not decorative attachments: their page-level extracted records are stored in `data/policy_documents.csv` and the cleaned database workbook.

## Production architecture
`User → Gemini query understanding → LangGraph → LangChain chunking/RAG → hybrid retrieval → deterministic verification → Gemini grounded response → validation gate → answer/sources`

## Important data-quality design
Course identity is separated from course-code availability. A course can exist even when the source course-code field says `DON’T KNOW`. Unknown/blank prerequisite information is never converted into “no prerequisite.”

## Deployment
1. Upload the repository to GitHub.
2. Deploy `app.py` from Streamlit Community Cloud.
3. Add a newly generated Gemini key to Streamlit Secrets:
```toml
GEMINI_API_KEY = "YOUR_NEW_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
```
Never commit the API key to GitHub.

## Demonstration queries
- `What is the prerequisite for Financial and Management Accounting?`
- `What is the prerequisite for Communication Skills?`
- `What is the attendance requirement?`
- `How long do I have to add or drop a course?`
- `Can I audit a course and still earn credits?`
- `Can SYN001 take UCOR205?`
- `What courses are in the Finance minor for 2022?`

# AIRA — AI Academic Advisor

## Final production deployment

This release fixes the LangGraph production startup/runtime issue and uses the following production architecture:

**User question → Gemini semantic understanding → LangGraph orchestration → LangChain recursive chunking → SentenceTransformer + FAISS retrieval → structured deterministic verification → Gemini grounded response generation → grounding validation → structured answer + source evidence**

### Authoritative academic sources
Only the two supplied Excel workbooks are authoritative academic sources:
- `data/Minor_Courses_for_BTech_Students(3).xlsx`
- `data/Semester_Spread_Structures_Sept_2026(3).xlsx`

`data/CLEANED_ACADEMIC_DATABASE.xlsx` is the normalized retrieval layer generated from those workbooks and retained for reproducibility/deployment consistency.

### Gemini configuration
Do **not** put the API key in GitHub or in `app.py`.
Configure Streamlit Cloud Secrets:

```toml
GEMINI_API_KEY = "YOUR_NEW_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
```

The app reads the key from Streamlit Secrets/environment. Gemini is used in production for semantic query understanding and final response generation, but it cannot override verified academic facts.

### LangGraph runtime safety
The production graph explicitly starts with the normalization node before intent understanding. If LangGraph encounters a runtime/orchestration error, the app falls back to the same grounded verification pipeline instead of exposing a traceback to the student.

### GitHub layout
```text
AI-Academic-Advisor_2.0/
├── app.py
├── README.md
├── requirements.txt
├── evaluation_cases.csv
└── data/
    ├── CLEANED_ACADEMIC_DATABASE.xlsx
    ├── Minor_Courses_for_BTech_Students(3).xlsx
    └── Semester_Spread_Structures_Sept_2026(3).xlsx
```

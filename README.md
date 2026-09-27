# AIRA — AI Academic Advisor

Production-oriented Streamlit academic decision-support application.

## Source coverage
The runtime uses **all supplied academic data categories**:
- Semester/Structure Excel workbook
- Minor Courses Excel workbook
- Student Handbook PDF
- Student SOP PDF
- Every packaged structured CSV
- Synthetic student profiles and course history

## Architecture
User → Streamlit → LangGraph → intent/entity resolution → hybrid RAG (SentenceTransformer + FAISS + TF-IDF reranking) → deterministic structured verification → guarded LangChain LLM explanation → final validation → answer + evidence.

The LLM is never the source of truth. Exact academic facts, conflicts, ambiguity and student-specific eligibility are protected by deterministic verification.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

For deployment, put `GEMINI_API_KEY` in Streamlit Community Cloud Secrets. `OPENAI_API_KEY` is supported as a fallback.

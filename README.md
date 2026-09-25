# AIRA — AI Academic Advisor | Final Assignment Submission

## Final architecture
User question → Gemini open-ended understanding → LangGraph orchestration → hybrid retrieval (structured database + LangChain chunked FAISS RAG) → deterministic verification → Gemini grounded response generation → grounding guard → structured answer + source evidence.

## Authoritative data
Academic facts come only from the two supplied Excel workbooks. `CLEANED_ACADEMIC_DATABASE.xlsx` is a normalized derivative that preserves source workbook/sheet/row provenance.

## Key behavior
- Understands natural/unstructured questions rather than requiring fixed templates.
- Handles combinations of course/minor, batch/year, semester, credits, prerequisites, hours, structure category, counts, lists, comparisons and student context.
- Detects ambiguity and missing context and asks a focused follow-up.
- Does not silently merge different academic batches/structures.
- Uses Gemini for query understanding and response generation.
- Uses LangChain text splitting and FAISS semantic retrieval.
- Uses LangGraph to orchestrate understand → retrieve/plan → verify → generate.
- Deterministic verification remains the factual authority.
- If the LLM output is unsupported or unavailable, the system falls back to a verified answer.

## Deployment
1. Push `app.py`, `requirements.txt`, `README.md`, `evaluation_cases.csv` and the `data/` folder to GitHub.
2. In Streamlit Secrets add:
```toml
GEMINI_API_KEY = "YOUR_NEW_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
```
3. Never commit an API key to GitHub.
4. Streamlit main file: `app.py`.

## Important
The included evaluation/test artifacts document the tested scenarios. No finite test suite can guarantee every possible English sentence; the design instead uses open-ended semantic planning plus safe deterministic execution.

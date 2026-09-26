# Assignment Rubric Alignment

| Phase | Evidence in submission | Status |
|---|---|---|
| 1 Build | Hybrid RAG, structured Excel database, source provenance, Gemini, LangChain chunking, FAISS, LangGraph, follow-up handling | Covered |
| 2 Challenge | 6 synthetic/anonymized profiles, 30 predefined cases, ambiguity, missing information, unknown course, insufficient student context | Covered |
| 3 Improve | Evidence-first prompt, constraints, JSON query plan, follow-up rules, structured student records, grounded response format | Covered |
| 4 Measure | 30 verified cases, automated retrieval/routing tests, accuracy/grounding/latency methodology, source correctness and uncertainty categories | Covered; final live latency depends on deployment |
| 5 Analyze | Basic LLM → Prompting → RAG → RAG + Structured Student Data comparison and limitations | Covered |
| 6 Deploy | Streamlit prototype, source cards, status indicators, Gemini Secrets, fallback path | Covered |
| 7 Document | 10-page report, PPT, notebook, architecture and testing documents | Covered |

## Data-scope limitation

Only the two supplied Excel workbooks are treated as authoritative application sources. The assignment description mentions additional possible documents such as a Student Handbook and SOP; those files were not supplied in the actual project data package, so the system does not claim to answer policies that are absent from the two supplied workbooks.

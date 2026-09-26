# Phase 5 — Compare and Analyze

The assignment requires comparison across four development stages. The comparison below is deliberately descriptive where a stable independent LLM baseline was not available; no unsupported benchmark score is claimed.

| Stage | Added capability | What it can demonstrate | Main limitation |
|---|---|---|---|
| Basic LLM / document Q&A | General LLM response over supplied text | Natural-language answers | Weak exact filtering, ambiguity handling and source verification |
| Structured Prompting | Role, context, constraints, delimiters, follow-up and output schema | More consistent and cautious responses | Prompting alone cannot reliably execute exact multi-condition academic queries |
| RAG | Retrieval of relevant academic evidence with provenance | Better grounding and paraphrase recall | Vector similarity alone is weak for exact counts and relational checks |
| RAG + Structured Student Data | Hybrid retrieval + normalized records + synthetic student history | Exact filters, eligibility checks, prerequisite verification and safer follow-up | Still limited by completeness/quality of supplied university data |

## Key finding

The main improvement comes from separating responsibilities. Gemini interprets natural language and explains verified evidence; LangGraph controls the workflow; LangChain/FAISS improve semantic recall; the normalized structured database performs exact filtering and verification. This directly supports the requirement that the advisor should know when to answer, when to ask, and when information is insufficient.

## Important evaluation note

The final 30-case evaluation measures the grounded final pipeline. It should not be presented as a pure head-to-head benchmark of four independent LLM systems. Where an independent baseline was not stable enough for reliable scoring, the comparison is qualitative rather than fabricated.

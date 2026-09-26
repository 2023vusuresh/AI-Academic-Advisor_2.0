# AIRA Architecture

## Offline
1. Load the two supplied Excel workbooks.
2. Normalize course codes, titles, batches, semesters, prerequisites and numeric fields.
3. Preserve source workbook, sheet and row metadata.
4. Build the normalized academic database.
5. Convert source records to RAG documents and split with LangChain `RecursiveCharacterTextSplitter`.
6. Embed chunks with SentenceTransformer and index them in FAISS.

## Online
1. User asks a natural-language question.
2. Gemini produces an open-ended semantic plan: relevant table(s), fields, filters, relationships, grouping, aggregation and output.
3. LangGraph routes the request through retrieval and verification nodes.
4. Structured retrieval applies exact metadata filters.
5. Vector RAG supplies semantic evidence for paraphrased questions.
6. Evidence is merged and reranked.
7. Deterministic verification checks counts, records, prerequisites, ambiguity and student context.
8. If context is insufficient, the graph asks a follow-up; if the requested fact is absent, it states that clearly.
9. Gemini generates a concise structured response from verified evidence.
10. A grounding guard prevents unsupported LLM claims from being shown.

## Why hybrid retrieval
Academic questions often require exact equality (e.g., batch=2024, credits=3) and also natural-language matching. Exact structured filtering is used for correctness; vector retrieval is used for semantic recall.

## Response generation and grounding guard
The production response layer uses Gemini with a fixed JSON response contract: direct answer, plain-language explanation, verified details, and note. The renderer turns that JSON into a consistent student-facing response. Before display, a grounding guard checks for unsupported course codes, exposed source placeholders, and omission of a salient verified count. If validation fails, AIRA falls back to the deterministic source-verified response.

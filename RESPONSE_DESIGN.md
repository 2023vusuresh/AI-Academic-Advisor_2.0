# AIRA Response Design and Accuracy Controls

## Goal
AIRA is designed for students and first-time users who may ask questions in informal, incomplete, or conversational language. The system separates **understanding the question** from **verifying academic facts** and from **explaining the answer**.

## Response contract
Every production response is generated into a fixed structure:

1. **Answer** — the direct answer first.
2. **In simple terms** — a short plain-language explanation for a user unfamiliar with university terminology.
3. **Details** — only useful facts supported by the verified result.
4. **Note** — scope, uncertainty, or limitation when relevant.

The UI shows source evidence separately with workbook, sheet and row provenance.

## Accuracy controls
- Gemini may interpret natural language but is not the academic source of truth.
- Structured Excel-derived records are filtered and verified before response generation.
- The final LLM response is checked for unsupported course codes and missing verified counts.
- Source placeholders such as `DON'T KNOW`, `UNKNOWN`, `TBD`, and `N/A` are never treated as real course codes.
- Conflicting academic versions are not silently merged.
- Missing information produces a follow-up or insufficient-information response.
- If Gemini is unavailable or returns an invalid structured response, AIRA uses a deterministic, source-grounded fallback.

## Natural-language coverage
The query-understanding layer is schema-driven rather than restricted to a small list of fixed question templates. It is designed to preserve combinations such as:

- minor + batch + semester + credits
- course + prerequisite
- course + dependent courses
- year-wise comparisons
- counts, lists, averages, totals and filtered records where the source fields support them
- conversational wording such as "hey AIRA, how many subjects are there for finance minor in 2024?"

## Important scope rule
AIRA answers only from the supplied academic records. If a question requires information that is not represented in those records, it says so rather than using outside knowledge.

# Testing and Validation

## Static validation
- `app.py` syntax checked with Python compile.
- ZIP/package contents verified.

## Recorded automated routing suite
- 220 generated/representative natural-language combinations.
- 220 valid routed responses.
- 0 execution errors.

## 30-case evaluation set
The repository includes `evaluation_cases.csv` with 30 predefined expected outcomes covering ambiguity, prerequisites, eligibility, offerings, minors, structures, filters, missing information and unknown data.

## Grounding principle
LLM-generated responses are constrained by verified records. When Gemini is unavailable or its response is not grounded, the system uses the deterministic verified result rather than inventing academic facts.

# AIRA — AI Academic Advisor | DATA308 Assignment #1

AIRA is a grounded academic decision-support prototype that determines whether a university question can be answered from the supplied academic source package, whether clarification is required, or whether the available evidence is insufficient.

## Authoritative source package

The repository contains the original source files:

- `data/Minor_Courses_for_BTech_Students(3).xlsx`
- `data/Semester_Spread_Structures_Sept_2026(3).xlsx`
- `data/Student_Handbook_August_2026.pdf`
- `data/SOP_STUDENT_17082026_FINAL.pdf`

The runtime does **not** repeatedly query these raw files. They are preserved for provenance and auditability.

## Runtime knowledge base

`data/CLEANED_ACADEMIC_DATABASE.xlsx` is the normalized structured knowledge base generated from the supplied academic sources.

It contains canonical representations for:

- course master
- course aliases
- prerequisites
- prerequisite relationships
- semester offerings
- minor courses
- programme structures
- degree requirements
- policy documents
- synthetic students
- student course history

`data/policy_documents.csv` stores page-level Handbook/SOP evidence for retrieval.

## Production architecture

```text
Source files
  → ingestion and normalization
  → canonical structured knowledge base
  → LangChain chunking
  → SentenceTransformer embeddings
  → FAISS index

User query
  → Gemini intent/entity understanding
  → LangGraph orchestration
  → entity resolution
  → hybrid structured + semantic retrieval
  → evidence reranking
  → deterministic verification
  → follow-up / insufficient / verified decision
  → Gemini grounded response
  → grounding validation
  → structured answer + source evidence
```

The structured database is the authority for academic facts. Gemini is used for language understanding and response generation, not for inventing university facts.

## Data-quality rules

- A source placeholder such as `DON'T KNOW` in a course-code field does not mean the course is absent.
- `NIL`/`None` is distinct from an unknown or unrecorded prerequisite.
- Multiple course identities with the same title are treated as ambiguity and trigger a specific clarification.
- Academic batches and structures are preserved as separate versions.
- Material conflicts are not silently overwritten.
- Student-specific eligibility uses the synthetic student profile and course history.

## Production UI

The student-facing interface contains only:

- student-profile selection
- natural-language question input
- structured answer
- status
- source evidence

The production interface does not expose an evaluation/faculty panel.

## Deployment

1. Push the repository to GitHub.
2. Deploy `app.py` using Streamlit Community Cloud.
3. Add a newly generated Gemini API key in Streamlit Secrets:

```toml
GEMINI_API_KEY = "YOUR_NEW_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
```

Never commit the API key to GitHub.

## Demonstration queries

```text
What is the prerequisite for Financial and Management Accounting?
What is the prerequisite for Communication Skills?
Which courses depend on Data Structures?
What is the attendance requirement?
How long do I have to add or drop a course?
Can I audit a course and still earn credits?
Can SYN001 take UCOR205?
What courses are in the Finance minor for 2022?
```

## Important limitation

AIRA is a decision-support prototype. It does not replace official registration, programme administration or Faculty Advisor decisions where the supplied source package does not determine the answer.

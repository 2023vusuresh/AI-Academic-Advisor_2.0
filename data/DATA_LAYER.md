# Data Layer

## Runtime files

`CLEANED_ACADEMIC_DATABASE.xlsx` is the normalized runtime knowledge base.

`course_aliases.csv` contains canonical course aliases.

`policy_documents.csv` contains page-level extracts from the two policy PDFs used by the RAG layer.

## Original source files

The original Excel workbooks and policy PDFs are retained in this directory for provenance, auditability and reproducibility.

They are **not** the primary runtime query layer. The application loads the cleaned structured database and the extracted policy table.

## Source hierarchy

1. Supplied university source files
2. Cleaned/canonical structured records
3. Retrieval evidence
4. Deterministic verification
5. Gemini response wording

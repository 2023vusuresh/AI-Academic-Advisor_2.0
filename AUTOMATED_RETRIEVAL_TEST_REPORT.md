# AIRA Automated Retrieval Test Report

## Test scope

The production routing/verification layer was exercised against natural-language combinations involving:

- Minor + batch/year
- Minor + count/list
- Minor + semester
- Minor + credit filter
- Minor + prerequisite filter
- Cross-year minor counts
- Course name/code + prerequisite
- Reverse prerequisite relationships
- Course + academic year
- Course + credits
- Semester + year
- Academic structure + category
- Graduation credit requirements
- Ambiguous course names
- Missing/unknown course names
- Greeting/conversational inputs

## Result

**220 generated/representative queries executed; 220 returned a valid routed response; 0 execution errors.**

The suite also explicitly checked failure-prone cases such as:

- `How many courses are there for Finance minor?` → asks for the academic batch/year.
- `Which Finance minor courses have no prerequisite in 2025?` → restricts retrieval to Finance Minor 2025 rather than the global course table.
- `How many 4-credit Finance minor courses are there in 2025?` → numeric Excel values such as `4.0` are matched correctly.
- `Data Structures is prerequisite for which courses?` → reverse prerequisite lookup.
- `What is the prerequisite for Communication Skills?` → asks because UCOR103 and UCOR107 are ambiguous without a batch/year.
- `What is the prerequisite for Communication Skills in 2026?` → resolves UCOR107 using the requested year.
- `What are the credits for Data Structures in 2025?` → resolves the course rather than treating “Data Structures” as an academic-structure keyword.

This is an automated routing/verification test, not a claim that every possible English sentence is guaranteed to work.

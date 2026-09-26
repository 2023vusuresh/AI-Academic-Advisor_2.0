
import os, re, time, math, hashlib, json
from pathlib import Path
from typing import TypedDict, Any, Dict, List
import pandas as pd
import streamlit as st
import openpyxl

# LangChain + LangGraph are used for document chunking and stateful orchestration.
# The local fallback keeps the app importable in environments where dependencies
# have not yet been installed; production requirements install the real packages.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except Exception:
    RecursiveCharacterTextSplitter = None
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = START = END = None
    LANGGRAPH_AVAILABLE = False

# ============================================================
# AIRA — AI Academic Advisor | Assignment #1
# Authoritative sources: the supplied academic package — two Excel workbooks plus the Student Handbook and SOP.
# RAG: Excel extraction -> normalization -> question routing/entity resolution ->
# 500/100 chunks ->
# SentenceTransformer embeddings -> FAISS retrieval -> rerank ->
# structured verification -> guarded LLM explanation.
# ============================================================

st.set_page_config(page_title="AIRA | Academic Advisor", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MINOR_FILE = DATA / "Minor_Courses_for_BTech_Students(3).xlsx"
STRUCT_FILE = DATA / "Semester_Spread_Structures_Sept_2026(3).xlsx"
HANDBOOK_FILE = DATA / "Student_Handbook_August_2026.pdf"
SOP_FILE = DATA / "SOP_STUDENT_17082026_FINAL.pdf"
POLICY_FILE = DATA / "policy_documents.csv"
ALIASES_FILE = DATA / "course_aliases.csv"

PASS_GRADES = {"A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-"}
FAIL_GRADES = {"F","FAIL","FAILED"}

# ---------- data cleaning & organization ----------
def norm(x):
    if x is None or (isinstance(x,float) and pd.isna(x)): return ""
    return re.sub(r"\s+"," ",str(x).replace("\xa0"," ").replace("’","'").strip())

def norm_code(x):
    """Normalize a real academic course code. Placeholder source values are NOT codes."""
    v = re.sub(r"\s+", "", norm(x).upper())
    # The source workbooks contain placeholders such as DON'T KNOW/TBD in the
    # course-code column. Never expose or treat those placeholders as course codes.
    if v.replace("’", "'") in {"DON'TKNOW", "DONTKNOW", "UNKNOWN", "NOTKNOWN", "NOTAVAILABLE", "NA", "N/A", "TBD", "NEW"}:
        return ""
    # Academic course codes in the supplied records follow LETTERS + 3 DIGITS.
    # Keeping only this shape prevents arbitrary source text from becoming a code.
    if not re.fullmatch(r"[A-Z]{2,8}\d{3}", v):
        return ""
    return v

def display_course_code(value):
    """Human-facing course-code label; never display source placeholders as codes."""
    code = norm_code(value)
    return code if code else "Course code not recorded"

def clean_semester(x):
    s=norm(x)
    if not s: return ""
    m=re.search(r"([1-8])",s)
    return f"S{m.group(1)}" if m else s

def clean_prerequisite(x):
    raw=norm(x)
    if not raw: return "", "not_recorded"
    u=raw.upper()
    if u in {"NIL","NONE","NO PREREQUISITE","NO PREREQUISITES"}:
        return "Nil", "explicit_none"
    if u in {"TBD","TO BE DECIDED","TO BE CONFIRMED"}:
        return raw, "pending"
    raw=re.sub(r"\s*[,;]\s*", ", ", raw)
    raw=re.sub(r"\s*/\s*", "/", raw)
    return raw, "recorded"

def batch_marker(x):
    s=norm(x)
    m=re.search(r"\b(20\d{2})\s*(?:batch|academic\s*year|year)?\b",s,re.I)
    return m.group(1) if m else ""

def extract_years(q):
    """Extract academic years from natural-language variants: 2025, batch 2025, 2025-26, AY2025, etc."""
    ql=norm(q).lower()
    years=[]
    for m in re.finditer(r"\b(?:ay|a\.y\.|academic\s+year|academic\s+batch|batch|year)?\s*(20\d{2})(?:\s*[-/]\s*(?:20)?\d{2})?\b",ql,re.I):
        y=m.group(1)
        if y not in years: years.append(y)
    return years

def _minor_year_intent(q):
    """Return (years, all_years) for minor questions without relying on one exact phrase."""
    years=extract_years(q)
    ql=norm(q).lower()
    all_years=any(x in ql for x in ["all years","all batches","every year","each year","year wise","year-wise","by year","across years","across all years","compare years","compare batches","batch wise","batch-wise"])
    return years,all_years

@st.cache_data(show_spinner=False)
def load_authoritative_data():
    """Load the normalized academic database produced from the two authoritative Excel workbooks.
    The original workbooks remain bundled for provenance; the normalized workbook is the
    application retrieval layer so repeated wide Excel layouts do not reach the retriever raw.
    """
    structured_file = DATA / "CLEANED_ACADEMIC_DATABASE.xlsx"
    if not structured_file.exists():
        # Backward-compatible fallback to the original parser if the normalized file is absent.
        return _load_original_authoritative_data()

    def read(sheet):
        return pd.read_excel(structured_file, sheet_name=sheet)

    minor = read("minor_courses").copy()
    minor["source_workbook"] = MINOR_FILE.name
    minor["source_type"] = "minor_excel"
    minor["source_sheet"] = minor["source_sheet"].astype(str)
    minor["batch"] = minor["batch"].astype(str).replace("nan", "")
    minor["course_code"] = minor["course_code"].fillna("").map(norm_code)
    minor["course_title"] = minor["course_title"].fillna("").map(norm)
    minor["semester"] = minor["semester"].fillna("").map(clean_semester)
    minor["prerequisite"] = minor["prerequisite"].fillna("").map(lambda x: clean_prerequisite(x)[0])

    sem = read("semester_courses").copy()
    sem["source_workbook"] = STRUCT_FILE.name
    sem["source_type"] = "semester_excel"
    sem["source_sheet"] = sem["source_sheet"].astype(str)
    sem["course_code"] = sem["course_code"].fillna("").map(norm_code)
    sem["course_title"] = sem["course_title"].fillna("").map(norm)
    sem["semester"] = sem["semester"].fillna("").map(clean_semester)
    sem["prerequisite"] = sem["prerequisite"].fillna("").map(lambda x: clean_prerequisite(x)[0])
    sem["batch"] = sem["academic_structure"].astype(str).str.extract(r"(20\d{2})", expand=False).fillna("")
    sem["category"] = sem["basket_name"].fillna("").map(norm)
    sem["basket"] = sem["basket_name"].fillna("").map(norm)

    structures = read("structure_courses").copy()
    structures["source_workbook"] = STRUCT_FILE.name
    structures["source_type"] = "structure_excel"
    structures["source_sheet"] = structures["academic_structure"].astype(str)
    structures["category"] = structures["structure_category"].fillna("").map(norm)
    structures["course_title"] = structures["course_title"].fillna("").map(norm)
    structures["credits"] = structures["credits"]
    structures["batch"] = structures["academic_structure"].astype(str).str.extract(r"(20\d{2})", expand=False).fillna("")

    req = read("degree_requirements").copy()
    req["source_workbook"] = STRUCT_FILE.name
    req["source_type"] = "structure_excel"
    req["source_sheet"] = req["academic_structure"].astype(str)
    req["basket"] = req["component"].fillna("").map(norm)
    req["credits"] = req["required_credits"]
    req["batch"] = req["academic_structure"].astype(str).str.extract(r"(20\d{2})", expand=False).fillna("")

    placeholders = pd.DataFrame()
    try: placeholders = pd.concat([read("semester_unknowns"), read("minor_unknowns")], ignore_index=True)
    except Exception: pass

    # Course master from the normalized prerequisite table gives one clean searchable
    # representation while preserving year-specific rows in semester/minor tables.
    course_master = read("course_master").copy()
    course_master["course_code"] = course_master["course_code"].fillna("").map(norm_code)
    course_master["course_title"] = course_master["course_title"].fillna("").map(norm)
    course_master["prerequisite"] = course_master["prerequisite"].fillna("").map(lambda x: clean_prerequisite(x)[0])

    # Build a unified course table from source-bearing rows only.
    course_parts=[]
    for _,r in sem[sem.course_title.astype(str).str.strip().ne("")].iterrows():
        course_parts.append({"course_code":r.course_code,"course_title":r.course_title,"credits":r.credits,
            "prerequisite":r.prerequisite,"prerequisite_status":r.prerequisite_status,"batch":r.batch,
            "source_type":"semester_excel","source_sheet":r.source_sheet,"source_row":r.source_row,
            "source_workbook":STRUCT_FILE.name,"semester":r.semester,"category":r.category,"basket":r.basket})
    for _,r in minor[minor.course_title.astype(str).str.strip().ne("")].iterrows():
        course_parts.append({"course_code":r.course_code,"course_title":r.course_title,"credits":r.credits,
            "prerequisite":r.prerequisite,"prerequisite_status":r.prerequisite_status,"batch":r.batch,
            "source_type":"minor_excel","source_sheet":r.source_sheet,"source_row":r.source_row,
            "source_workbook":MINOR_FILE.name,"semester":r.semester,"category":"","basket":""})
    # The prerequisite/course-master table contains valid course identities that may not
    # appear in the semester spread for every batch. Keep them for entity resolution.
    for _,r in course_master[course_master.course_title.astype(str).str.strip().ne("")].iterrows():
        course_parts.append({"course_code":r.course_code,"course_title":r.course_title,"credits":"",
            "prerequisite":r.prerequisite,"prerequisite_status":"","batch":"",
            "source_type":"course_master","source_sheet":"Course Master","source_row":"",
            "source_workbook":STRUCT_FILE.name,"semester":"","category":"","basket":""})
    courses=pd.DataFrame(course_parts).drop_duplicates(subset=["course_code","course_title","prerequisite","source_type"])

    return minor,sem,structures,req,courses,placeholders

# Preserve the previous parser as a fallback without changing its behavior.
_original_loader_start = None

def _load_original_authoritative_data():
    """Fallback parser copied from the earlier release; used only if normalized DB is absent."""
    # The deployment package always includes CLEANED_ACADEMIC_DATABASE.xlsx, so this
    # fallback is intentionally compact and raises a useful message if it is missing.
    raise FileNotFoundError("CLEANED_ACADEMIC_DATABASE.xlsx is missing from data/. Add the normalized academic database generated from the two supplied Excel workbooks.")

# ---------- synthetic student data ----------
@st.cache_data
def synthetic_students():
    students=pd.DataFrame([
        ["SYN001","BTech","2026",5,72,"Finance"],
        ["SYN002","BTech","2026",6,96,"Psychology"],
        ["SYN003","BTech","2025",7,120,"Finance"],
        ["SYN004","BTech","2026",5,72,"Finance"],
        ["SYN005","BTech","2025",7,118,"Marketing"],
        ["SYN006","BTech","2026",6,94,"Economics"],
    ],columns=["student_id","programme","batch","semester","completed_credits","minor"])
    history=pd.DataFrame([
        ["SYN001","UCOR103","Passed","A"],["SYN001","UCOR203","Passed","B"],
        ["SYN001","UCOR205","Failed","F"],["SYN001","UCOR104","Passed","A"],
        ["SYN002","UCOR103","Passed","A"],["SYN002","UCOR203","Passed","B"],
        ["SYN002","UCOR104","Passed","B+"],
        ["SYN003","UCOR103","Passed","A"],["SYN003","UCOR203","Passed","A"],
        ["SYN003","UCOR205","Passed","B"],["SYN003","UCOR310","Passed","A-"],
        ["SYN004","UCOR103","Passed","A"],
        ["SYN005","UCOR103","Passed","B"],["SYN006","UCOR103","Passed","A"],
    ],columns=["student_id","course_code","status","grade"])
    return students,history

# ---------- RAG corpus ----------
def record_to_text(row):
    parts=[]
    for k,v in row.items():
        if norm(v): parts.append(f"{k}: {norm(v)}")
    return " | ".join(parts)

def tokenize_words(s):
    return re.findall(r"\b[\w&'-]+\b", str(s).lower())

def chunk_text(text, meta, target_tokens=500, overlap=100):
    """Chunk a normalized academic record with LangChain's recursive splitter.

    Academic records are already compact, so most remain one chunk. Longer
    records are split on natural separators rather than arbitrary token cuts.
    Metadata is retained on every chunk for source-aware retrieval.
    """
    text = str(text)
    if LANGCHAIN_AVAILABLE and RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1600,
            chunk_overlap=250,
            separators=["\n", " | ", "; ", ", ", " ", ""],
            length_function=len,
        )
        docs = splitter.create_documents([text], metadatas=[dict(meta)])
        return [dict(d.metadata, text=d.page_content, chunk_index=i) for i,d in enumerate(docs)]
    # Fallback only for local environments missing LangChain.
    toks=text.split()
    if len(toks)<=target_tokens:
        return [{"text":text,**meta,"chunk_index":0}]
    out=[]; start=0; idx=0
    step=max(1,target_tokens-overlap)
    while start<len(toks):
        end=min(len(toks),start+target_tokens)
        out.append({"text":" ".join(toks[start:end]),**meta,"chunk_index":idx})
        idx+=1
        if end==len(toks): break
        start+=step
    return out

@st.cache_data(show_spinner=False)
def load_policy_documents():
    if not POLICY_FILE.exists():
        return pd.DataFrame(columns=["document","page","section","text","source_file"])
    d=pd.read_csv(POLICY_FILE).fillna("")
    d["page"]=pd.to_numeric(d["page"],errors="coerce").fillna(0).astype(int)
    return d

@st.cache_data(show_spinner=False)
def load_course_aliases():
    if not ALIASES_FILE.exists():
        return pd.DataFrame(columns=["course_id","alias","normalized_alias","canonical_title","course_code","source_tables"])
    return pd.read_csv(ALIASES_FILE).fillna("")

@st.cache_resource(show_spinner="Preparing academic information…")
def build_rag():
    minors, semesters, structures, requirements, courses, placeholders = load_authoritative_data()
    policy=load_policy_documents()
    aliases=load_course_aliases()
    records=[]
    for _,r in minors.iterrows():
        records.append((record_to_text(r.to_dict()),{"record_type":"minor_course","source":"authoritative_excel","source_workbook":MINOR_FILE.name,"source_sheet":r.get("source_sheet","") ,"source_row":r.get("source_row","")}))
    for _,r in semesters.iterrows():
        records.append((record_to_text(r.to_dict()),{"record_type":"semester_course","source":"authoritative_excel","source_workbook":STRUCT_FILE.name,"source_sheet":r.get("source_sheet","") ,"source_row":r.get("source_row","")}))
    for _,r in structures.iterrows():
        records.append((record_to_text(r.to_dict()),{"record_type":"programme_structure","source":"authoritative_excel","source_workbook":STRUCT_FILE.name,"source_sheet":r.get("source_sheet","") ,"source_row":r.get("source_row","")}))

    chunks=[]
    for i,(t,meta) in enumerate(records):
        chunks.extend(chunk_text(t,{**meta,"record_id":i}))
    # Policy documents are first-class RAG records. Each page is chunked while preserving PDF/page provenance.
    for _,r in policy.iterrows():
        t=f"DOCUMENT: {r.document}\nPAGE: {r.page}\nSECTION: {r.section}\nCONTENT: {r.text}"
        chunks.extend(chunk_text(t,{"record_type":"policy_document","source":r.document,"source_workbook":r.source_file,"source_sheet":r.section,"source_row":r.page,"page":r.page,"document":r.document}))

    texts=[c["text"] for c in chunks]
    # Primary RAG: dense embeddings + FAISS.
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        model=SentenceTransformer("all-MiniLM-L6-v2")
        emb=model.encode(texts,normalize_embeddings=True,show_progress_bar=False)
        emb=emb.astype("float32")
        index=faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
        return {"mode":"dense_faiss","chunks":chunks,"texts":texts,"model":model,"index":index,
                "minors":minors,"semesters":semesters,"structures":structures,"requirements":requirements,"courses":courses,"placeholders":placeholders,"policy_documents":policy,"course_aliases":aliases}
    except Exception as e:
        # Explicitly labelled fallback so the system never pretends TF-IDF is dense RAG.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
        X=vec.fit_transform(texts)
        return {"mode":"lexical_fallback","chunks":chunks,"texts":texts,"vectorizer":vec,"X":X,
                "error":str(e),"minors":minors,"semesters":semesters,"structures":structures,"requirements":requirements,"courses":courses,"placeholders":placeholders,"policy_documents":policy,"course_aliases":aliases}

def retrieve(query, rag, k_candidates=40, k_final=12):
    q=norm(query)
    # query processing / expansion
    expanded=q
    synonyms={"pre requisite":"prerequisite","pre-requisite":"prerequisite",
              "enroll":"eligibility","register":"eligibility","subject":"course",
              "subjects":"courses","available":"offered"}
    for a,b in synonyms.items(): expanded=expanded.replace(a,b)
    codes=re.findall(r"\b[A-Za-z]{2,8}\d{3}\b",q.upper())

    if rag["mode"]=="dense_faiss":
        qv=rag["model"].encode([expanded],normalize_embeddings=True).astype("float32")
        scores,ids=rag["index"].search(qv,k_candidates)
        cand=[(int(i),float(s)) for i,s in zip(ids[0],scores[0]) if i>=0]
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        qv=rag["vectorizer"].transform([expanded])
        scores=cosine_similarity(qv,rag["X"]).ravel()
        ids=scores.argsort()[::-1][:k_candidates]
        cand=[(int(i),float(scores[i])) for i in ids]

    qwords=set(tokenize_words(expanded))
    reranked=[]
    for i,sem_score in cand:
        txt=rag["texts"][i]
        words=set(tokenize_words(txt))
        lexical=len(qwords & words)/max(1,len(qwords))
        bonus=0.0
        for code in codes:
            if code in txt.upper(): bonus+=0.20
        title_terms=[w for w in qwords if len(w)>2]
        title_overlap=sum(1 for w in title_terms if w in txt.lower())/max(1,len(title_terms))
        final=0.72*sem_score+0.18*lexical+0.10*title_overlap+bonus
        reranked.append((final,i,sem_score,lexical))
    reranked.sort(reverse=True)
    return [rag["chunks"][i] | {"score":s,"semantic_score":ss,"lexical_score":ls}
            for s,i,ss,ls in reranked[:k_final]]

# ---------- entity resolution / deterministic verification ----------
def _clean_title(s):
    return re.sub(r"[^a-z0-9]+", " ", norm(s).lower()).strip()

def _query_words(q):
    stop={"what","which","when","where","who","how","does","do","is","are","the","a","an","for","of","to","in","on","and","or","can","i","my","me","please","tell","give","show","list","course","courses","about","with","from","this","that","does","it","its"}
    return [w for w in re.findall(r"[a-z0-9]+", q.lower()) if len(w)>1 and w not in stop]

def find_course(query, courses):
    """Resolve course codes OR course names. Names are first-class user input."""
    q=norm(query)
    qcode=norm_code(q)
    exact_code=courses[courses.course_code.eq(qcode)] if qcode else courses.iloc[0:0]
    if len(exact_code)>0:
        # The same code can have harmless punctuation/wording differences across
        # academic-year sheets. Collapse near-identical titles; only true title
        # differences become an ambiguity.
        titles=list(exact_code.course_title.astype(str).map(_clean_title))
        base=set(titles[0].split()) if titles else set()
        near=[]
        for t in titles:
            ts=set(t.split()); sim=len(base & ts)/max(1,len(base | ts))
            if sim>=0.75: near.append(t)
        if len(near)==len(titles): return "exact",exact_code.iloc[0]
        return "ambiguous",exact_code.drop_duplicates(subset=["course_code","course_title"])
    codes=re.findall(r"\b[A-Za-z]{2,8}\d{3}\b",q.upper())
    for code in codes:
        hit=courses[courses.course_code.eq(norm_code(code))]
        if len(hit)>0:
            titles=list(hit.course_title.astype(str).map(_clean_title)); base=set(titles[0].split()) if titles else set()
            if all(len(base & set(t.split()))/max(1,len(base | set(t.split())))>=0.75 for t in titles):
                return "exact",hit.iloc[0]
            return "ambiguous",hit.drop_duplicates(subset=["course_code","course_title"])

    # Canonical alias lookup: abbreviations such as FAMA are first-class entities.
    try:
        aliases=rag.get("course_aliases") if "rag" in globals() else None
    except Exception:
        aliases=None
    if aliases is not None and len(aliases):
        nqt=_clean_title(q)
        ah=aliases[aliases.normalized_alias.eq(nqt)]
        if len(ah):
            candidates=[]
            for _,a in ah.iterrows():
                ct=_clean_title(a.get("canonical_title",""))
                hit=courses[courses.course_title.astype(str).map(_clean_title).map(lambda t: t==ct or t.startswith(ct+" ") or ct in t)]
                if len(hit):
                    unique=hit.drop_duplicates(subset=["course_code","course_title"])
                    if len(unique)>1:
                        return "ambiguous", unique
                    candidates.append(unique.iloc[0])
            if candidates:
                return "title", candidates[0]

    # Exact title or title phrase embedded in the user's natural-language question.
    qt=_clean_title(q)
    title_norm=courses.course_title.map(_clean_title)
    exact_title=courses[title_norm.eq(qt)]
    if len(exact_title)==0:
        embedded=courses[title_norm.map(lambda t: bool(t) and (t in qt or qt in t))]
        exact_title=embedded
    exact_title=exact_title.drop_duplicates(subset=["course_code","course_title"])
    if len(exact_title)==1: return "title",exact_title.iloc[0]
    if len(exact_title)>1: return "ambiguous",exact_title

    words=[w for w in _query_words(q) if w not in {"prerequisite","prereq","credit","credits","offered","offering","semester","eligible","eligibility","register","registration","enroll","enrol","take","available","availability","information","details","detail"}]
    if not words: return "none",None
    scored=[]
    for _,r in courses.drop_duplicates(subset=["course_code","course_title"]).iterrows():
        title_words=set(_clean_title(r.course_title).split())
        overlap=sum(w in title_words for w in words)/max(1,len(words))
        phrase=1.0 if all(w in _clean_title(r.course_title).split() for w in words) else 0.0
        score=0.75*overlap+0.25*phrase
        if score>0: scored.append((score,r))
    scored.sort(key=lambda x:x[0],reverse=True)
    if not scored or scored[0][0]<0.55: return "none",None
    top=scored[0][0]
    ties=[r for s,r in scored if s>=max(0.55,top-0.08)]
    # Only ask a follow-up when the leading matches are genuinely close.
    if len(ties)>1 and top < 1.0:
        return "ambiguous",pd.DataFrame(ties[:8])
    return "title",scored[0][1]

def _course_rows(courses, code, title=None):
    c=norm_code(code)
    if c:
        return courses[courses.course_code.eq(c)].drop_duplicates()
    if title:
        nt=_clean_title(title)
        return courses[courses.course_title.astype(str).map(_clean_title).eq(nt)].drop_duplicates()
    return courses.iloc[0:0].copy()

def _source_evidence(rows, label=None):
    out=[]
    for _,r in rows.iterrows():
        source=r.get("source_sheet",""); row=r.get("source_row","")
        stype=norm(r.get("source_type", ""))
        raw_workbook=norm(r.get("source_workbook", ""))
        workbook = ("Minor Courses workbook" if "Minor_Courses" in raw_workbook or stype == "minor_excel" else ("Semester & Structure workbook" if "Semester_Spread" in raw_workbook or stype in {"semester_excel","structure_excel"} else "University Excel source"))
        text=[]
        for col in ["course_code","course_title","credits","prerequisite","semester","minor","batch","lecture_hours","tutorial_hours","practical_hours","category","basket"]:
            if col in r.index and norm(r[col]): text.append(f"{col}: {norm(r[col])}")
        out.append({"source":source,"row":row,"workbook":workbook,"text":("; ".join(text) if text else norm(label or "authoritative Excel record"))})
    return out

def _is_list_query(q):
    return any(x in q for x in ["list all","list the","what courses","which courses","courses in","subjects in","show all","give me all"])

def _minor_match(q, minors):
    ql=q.lower()
    for m in sorted(minors.minor.dropna().unique(), key=len, reverse=True):
        if str(m).lower() in ql: return str(m)
    return None

def _semester_number(q):
    m=re.search(r"\b(?:semester|sem)\s*([1-8])\b",q.lower())
    return f"S{m.group(1)}" if m else None

def _structure_match(q, structures):
    ql=q.lower()
    sheets=[s for s in structures.source_sheet.dropna().unique() if str(s).lower() in ql]
    if sheets: return sheets[0]
    for token in ["2022","2023","2024","2025","2026"]:
        if token in ql:
            hit=structures[structures.source_sheet.astype(str).str.contains(token,case=False,na=False)]
            if len(hit): return hit.source_sheet.iloc[0]
    return None

def _course_fact_answer(code, title, courses, semesters, q):
    rows=_course_rows(courses,code)
    if rows.empty: return None
    # Preserve all distinct authoritative values instead of silently choosing one.
    titles=sorted(set(norm(x) for x in rows.course_title if norm(x)))
    credits=sorted(set(norm(x) for x in rows.credits if norm(x)))
    pres_raw=[norm(x) for x in rows.prerequisite if norm(x)]
    pres=[]
    for x in pres_raw:
        if x.upper() in {"NIL","NONE","NA","N/A","-"}: continue
        if x not in pres: pres.append(x)
    if not pres and pres_raw: pres=["Nil/none recorded"]
    evidence=_source_evidence(rows)
    wants_pre="prereq" in q or "pre req" in q or "pre-requisite" in q
    wants_credit=any(x in q for x in ["credit","credits","how many credit"])

    # Do not merge different academic-year/batch values into one misleading answer.
    if wants_pre:
        pre_pairs=[]
        for _,rr in rows.iterrows():
            pv=norm(rr.get("prerequisite","")) or "Not recorded"
            ver=norm(rr.get("batch","")) or norm(rr.get("source_sheet",""))
            item=(ver,pv)
            if item not in pre_pairs: pre_pairs.append(item)
        pre_values=sorted(set(v for _,v in pre_pairs))
        if len(pre_values)>1:
            detail="; ".join(f"{v}: {p}" for v,p in pre_pairs)
            return {"answer":f"The supplied records show different prerequisite entries for **{code} — {title}** across academic versions: {detail}. Please tell me the applicable batch/year so I can give the correct version.","status":"follow_up","evidence":evidence}

    if wants_credit:
        credit_pairs=[]
        for _,rr in rows.iterrows():
            cv=norm(rr.get("credits","")) or "Not recorded"
            ver=norm(rr.get("batch","")) or norm(rr.get("source_sheet",""))
            item=(ver,cv)
            if item not in credit_pairs: credit_pairs.append(item)
        credit_values=sorted(set(v for _,v in credit_pairs))
        if len(credit_values)>1:
            detail="; ".join(f"{v}: {c} credits" for v,c in credit_pairs)
            return {"answer":f"The supplied records show different credit values for **{code} — {title}** across academic versions: {detail}. Please tell me the applicable batch/year.","status":"follow_up","evidence":evidence}
    if wants_pre and wants_credit:
        val="; ".join(pres) if pres else "Nil/none recorded"
        cval=", ".join(credits) if credits else "not recorded"
        return {"answer":f"{code} — {title}: prerequisite **{val}**; credits **{cval}**.","status":"verified","evidence":evidence}
    if wants_pre:
        val="; ".join(pres) if pres else "Nil/none recorded"
        return {"answer":f"{code} — {title}: the prerequisite recorded in the supplied Excel data is **{val}**.","status":"verified","evidence":evidence}
    if wants_credit:
        val=", ".join(credits) if credits else "not recorded"
        return {"answer":f"{code} — {title}: credits recorded in the supplied Excel data: **{val}**.","status":"verified","evidence":evidence}
    if any(x in q for x in ["all details","all information","everything about","tell me about","details about","information about"]):
        items=[]
        if titles: items.append("Course title: " + "; ".join(titles))
        if credits: items.append("Credits: " + "; ".join(credits))
        if pres: items.append("Prerequisite: " + "; ".join(pres))
        else: items.append("Prerequisite: Nil/none recorded")
        rows2=semesters[semesters.course_code.astype(str).str.upper().eq(code.upper())].drop_duplicates()
        if not rows2.empty:
            items.append("Semester offerings: " + ", ".join(sorted(set(norm(x) for x in rows2.semester))))
            cats=sorted(set(norm(x) for x in rows2.category if norm(x)))
            if cats: items.append("Categories: " + "; ".join(cats))
            baskets=sorted(set(norm(x) for x in rows2.basket if norm(x)))
            if baskets: items.append("Baskets: " + "; ".join(baskets))
            evidence += _source_evidence(rows2)
        return {"answer":f"**{code} — {title}**\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":evidence}
    if any(x in q for x in ["semester and category","category and semester","semester/category","category/semester"]):
        rows2=semesters[semesters.course_code.astype(str).str.upper().eq(code.upper())].drop_duplicates()
        if rows2.empty:
            return {"answer":f"No semester-spread record was found for **{code} — {title}** in the supplied workbook.","status":"insufficient","evidence":evidence}
        sems=sorted(set(norm(x) for x in rows2.semester if norm(x)))
        cats=sorted(set(norm(x) for x in rows2.category if norm(x)))
        evidence += _source_evidence(rows2)
        return {"answer":f"**{code} — {title}** is listed in semester(s): **{', '.join(sems) if sems else 'not recorded'}**.\n\nCategory: **{'; '.join(cats) if cats else 'not recorded'}**.","status":"verified","evidence":evidence}
    if any(x in q for x in ["offer","available","when is","which semester","semester is","offered"]):
        rows2=semesters[semesters.course_code.str.upper().eq(code.upper())]
        if rows2.empty:
            return {"answer":f"{code} — {title} is present in the course data, but I do not have a semester-offering record for it in the supplied workbook.","status":"insufficient","evidence":evidence}
        offs=rows2[["source_sheet","source_row","semester","course_title","credits"]].drop_duplicates()
        sems=sorted(set(norm(x) for x in offs.semester))
        evidence += _source_evidence(offs)
        return {"answer":f"{display_course_code(code)} — {title} is listed for: **{', '.join(sems)}** in the supplied semester-spread workbook. This is a source listing, not a guarantee of future registration availability.","status":"verified","evidence":evidence}
    return None


# ---------- comprehensive database question router ----------
def _wants_list(q):
    return any(x in q.lower() for x in ["list", "show", "which", "what courses", "give me", "all courses", "all the courses", "how many courses"])

def _format_rows(rows, fields, limit=80):
    out=[]
    for _,r in rows.head(limit).iterrows():
        vals=[]
        for f,label in fields:
            if f in r.index and norm(r[f]):
                value=norm(r[f])
                if f=="course_code" and value.upper() in {"DON'T KNOW","DONT KNOW","DONTKNOW"}: value="not recorded"
                vals.append(f"{label}: {value}")
        if vals: out.append(" • ".join(vals))
    return out



def _prereq_query_type(q):
    ql=_clean_title(q)
    # IMPORTANT: “Which course is a prerequisite for X?” is a FORWARD
    # question (find the prerequisites of X). Do not confuse it with
    # “Which courses is X a prerequisite for?” (reverse relationship).
    reverse_patterns=[
        r"\bwhich\s+courses?\s+(?:depend|rel[y]?|rely)\s+on\b",
        r"\bwhat\s+courses?\s+(?:depend|rel[y]?|rely)\s+on\b",
        r"\bwhich\s+courses?\s+require\b",
        r"\bwhat\s+courses?\s+require\b",
        r"\bwhich\s+courses?\s+need\b",
        r"\bwhat\s+courses?\s+need\b",
        r"\bis\s+(?:a\s+)?prerequisite\s+for\s+which\s+courses?\b",
        r"\bare\s+prerequisites\s+for\s+which\s+courses?\b",
        r"\bis\s+(?:a\s+)?prerequisite\s+for\s+what\s+courses?\b",
        r"\b(?:which|what)\s+courses?\s+.+?\s+is\s+(?:a\s+)?prerequisite\s+for\b",
        r"\b(?:which|what)\s+courses?\s+(?:is|are)\s+.+?\s+(?:a\s+)?prerequisite\s+for\b",
    ]
    forward_patterns=[
        r"\bwhich\s+courses?\s+(?:is|are)\s+(?:a\s+)?prerequisites?\s+for\b",
        r"\bwhat\s+courses?\s+(?:is|are)\s+(?:a\s+)?prerequisites?\s+for\b",
        r"\bwhat\s+do\s+i\s+need\s+(?:to\s+)?(?:do|take|finish)?\s*before\b",
        r"\bwhat\s+do\s+i\s+need\s+(?:first|beforehand)\b",
        r"\bwhat\s+(?:subjects|courses|things)?\s*should\s+i\s+(?:take|finish|complete)\s+before\b",
        r"\bwhat\s+do\s+i\s+take\s+before\b",
        r"\bwhat\s+course\s+do\s+i\s+need\s+before\b",
        r"\bwhat\s+course\s+is\s+needed\s+before\b",
        r"\brequired\s+before\b",
        r"\bbefore\s+(?:i\s+can\s+)?take\b",
        r"\bbefore\s+taking\b",
        r"\bwhat\s+comes\s+before\b",
        r"\bprerequisites?\s+for\b",
        r"\bpre\s*requisites?\s+for\b",
    ]
    if any(re.search(p, ql) for p in reverse_patterns): return "reverse"
    if any(re.search(p, ql) for p in forward_patterns): return "forward"
    if "prerequisite" in ql or "prereq" in ql or "pre req" in ql:
        return "forward"
    return None

def _extract_target_phrase(q):
    ql=_clean_title(q)
    # Forward: target follows the “for/before” marker. Keep this deliberately
    # conservative so words such as “next semester” are not treated as course names.
    markers=[
        "prerequisite for ", "prerequisites for ", "prereq for ",
        "required before ", "before taking ", "before i can take ",
        "before i take ", "before taking the ", "before ",
    ]
    for marker in markers:
        if marker in ql:
            part=ql.split(marker,1)[1].strip(" ?.,!")
            part=re.split(r"\b(?:next|this|that)\s+semester\b", part, maxsplit=1)[0].strip()
            part=re.sub(r"\b(this|that|the|course|subject|module|class)\b$", "", part).strip()
            if part and part not in {"this","that","it","the course","the subject","the module"}:
                return part
    # Reverse: target follows “depend on / require / need”.
    for marker in ["depend on ", "depends on ", "require ", "requires "]:
        if marker in ql:
            part=ql.split(marker,1)[1].strip(" ?.,!")
            if part: return part
    # “X is a prerequisite for which courses?” — target is X.
    m=re.search(r"(?:which|what)\s+courses?\s+(?:is|are)\s+(.+?)\s+(?:a\s+)?prerequisites?\s+for", ql)
    if m: return m.group(1).strip(" ?.,!")
    m=re.search(r"(?:which|what)\s+courses?\s+(.*?)\s+(?:is|are)\s+(?:a\s+)?prerequisites?\s+for", ql)
    if m: return m.group(1).strip(" ?.,!")
    # If the wording says “I want to take X ... what do I need first?”,
    # the course entity can be resolved directly from the whole question.
    return ""

def _resolve_target(target, courses):
    if not target: return "none", None
    kind, course=find_course(target,courses)
    return kind, course

def prerequisite_relationship_answer(query, courses):
    """Answer both directions: prerequisites of a course and courses that depend on it."""
    mode=_prereq_query_type(query)
    if not mode: return None
    target=_extract_target_phrase(query)
    # Natural student wording often places the course entity elsewhere in the
    # sentence, e.g. “I want to take Internet of Things next sem — what do I need first?”
    if not target:
        fallback_kind, fallback_course=find_course(query,courses)
        if fallback_kind in {"exact","title"} and fallback_course is not None:
            kind, course=fallback_kind, fallback_course
        else:
            return {"answer":"Which course do you mean? Please give the course name or code.","status":"follow_up","evidence":[]}
    else:
        kind, course=_resolve_target(target,courses)
    if kind=="ambiguous":
        rows=course.drop_duplicates(subset=["course_code","course_title"])
        choices=", ".join(f"{display_course_code(r.course_code)} — {norm(r.course_title)}" for _,r in rows.iterrows())
        return {"answer":f"I found more than one match: **{choices}**. Which one do you mean?","status":"follow_up","evidence":_source_evidence(rows)}
    if kind=="none":
        return {"answer":f"I could not match **{target}** to a course in the supplied academic records. Please give the course code or full course name.","status":"follow_up","evidence":[]}

    target_code=norm_code(course.course_code); target_title=norm(course.course_title)
    target_rows=_course_rows(courses,target_code,target_title)
    if mode=="forward":
        # Forward lookup: what is required before taking the target course?
        prereq_values=[]; ev=[]
        for _,r in target_rows.iterrows():
            p=norm(r.get("prerequisite",""))
            if p and p not in prereq_values: prereq_values.append(p)
        ev=_source_evidence(target_rows)
        statuses=[]
        for _,rr in target_rows.iterrows():
            st=norm(rr.get("prerequisite_status",""))
            if st and st not in statuses: statuses.append(st)
        # Distinguish an explicit NIL/None from a blank/unknown prerequisite field.
        if not prereq_values:
            if any(st in {"unknown","not_recorded","pending"} for st in statuses) or not statuses:
                label = f"{display_course_code(target_code)} — {target_title}" if target_code else target_title
                return {"answer":f"I found **{label}** in the supplied academic records, but its prerequisite is **not specified in the source data**. I will not assume that this means there is no prerequisite.","status":"insufficient","evidence":ev}
        if all(p.upper() in {"NIL","NONE","NA","N/A","-"} for p in prereq_values):
            label = f"{display_course_code(target_code)} — {target_title}" if target_code else target_title
            return {"answer":f"**{label}** has **no prerequisite recorded** in the supplied academic data.","status":"verified","evidence":ev}
        # Preserve academic-version differences.
        if len(set(prereq_values))>1:
            detail="; ".join(prereq_values)
            return {"answer":f"The supplied records show different prerequisites for **{target_code} — {target_title}**: **{detail}**. Tell me the applicable batch/year and I’ll use that version.","status":"follow_up","evidence":ev}
        p=prereq_values[0]
        # Add course names for referenced codes where possible.
        refs=[]
        for rc in prerequisite_tokens(p):
            rr=courses[courses.course_code.eq(rc)].drop_duplicates(subset=["course_code","course_title"])
            if len(rr): refs.append(f"{rc} — {norm(rr.iloc[0].course_title)}")
            else: refs.append(rc)
        display="; ".join(refs) if refs else p
        return {"answer":f"To take **{target_code} — {target_title}**, the prerequisite recorded in the supplied academic data is **{display}**.","status":"verified","evidence":ev}

    # Reverse lookup: which course(s) list the target as a prerequisite?
    dependents=[]
    target_title_norm=_clean_title(target_title)
    for _,r in courses.iterrows():
        p=norm(r.get("prerequisite",""))
        if not p: continue
        refs=[norm_code(x) for x in re.findall(r"[A-Za-z]{2,8}\d{3}",p)]
        matched=target_code in refs
        if not matched and target_title_norm:
            pnorm=_clean_title(p)
            matched=target_title_norm in pnorm
        if matched:
            dependents.append(r)
    if not dependents:
        label=f"{display_course_code(target_code)} — {target_title}" if target_code else target_title
        return {"answer":f"I could not find any supplied course record that lists **{label}** as a prerequisite.","status":"verified","evidence":target_rows.head(8).to_dict("records") and _source_evidence(target_rows) or []}
    depdf=pd.DataFrame(dependents).drop_duplicates(subset=["course_code","course_title"])
    items=[]
    ev=[]
    for _,r in depdf.iterrows():
        items.append(f"**{display_course_code(r.course_code)} — {norm(r.course_title)}**")
        ev.append({"source":r.get("source_sheet",""),"row":r.get("source_row",""),"workbook":r.get("source_workbook","Semester & Structure workbook"),"text":f"{display_course_code(r.course_code)} — {norm(r.course_title)}; prerequisite: {norm(r.prerequisite)}"})
    verb="is" if len(items)==1 else "are"
    return {"answer":f"**{target_code} — {target_title}** {verb} listed as a prerequisite for:\n\n"+"\n".join(f"• {x}" for x in items),"status":"verified","evidence":ev}



# ---------- Generic LLM-to-structured-data planning ----------
GENERIC_SCHEMA = {
    "minor_courses": ["minor","batch","course_code","course_title","lecture_hours","tutorial_hours","practical_hours","credits","semester","prerequisite","prerequisite_status"],
    "semester_courses": ["academic_structure","semester","basket_code","basket_name","basket_fixed_min_credits","course_code","course_title","prerequisite","lecture_hours","tutorial_hours","practical_hours","credits","prerequisite_status"],
    "semester_offerings": ["academic_structure","semester","course_code","course_title","prerequisite","prerequisite_status","credits","basket_code","basket_name"],
    "structure_courses": ["academic_structure","structure_category","course_number","course_title","credits"],
    "degree_requirements": ["academic_structure","component","required_credits"],
    "course_master": ["course_code","course_title","prerequisite","has_source_conflict"],
    "prerequisites": ["course_code","course_title","prerequisite","prerequisite_status","prerequisite_codes"],
    "course_relationships": ["source_course","relationship","target_course","source_title","source_prerequisite_text"],
    "synthetic_students": ["student_id","batch","programme","current_semester","total_credits","minor"],
    "student_course_history": ["student_id","course_code","status","grade","course_exists"],
    "policy_documents": ["document","page","section","text","source_file"],
}


def gemini_generic_plan(query, student_id=None):
    # Open-ended semantic parser; deliberately schema-driven, not phrase-driven.
    schema_text="\n".join(f"- {k}: {', '.join(v)}" for k,v in GENERIC_SCHEMA.items())
    prompt=f'''You are the semantic planning engine for AIRA. The user may ask ANY natural-language question that can be answered from the supplied academic database. The wording may be informal, misspelled, indirect, multi-part, conversational, or contain several filters at once. Do not restrict yourself to a predefined intent list.

AVAILABLE TABLES AND FIELDS:
{schema_text}

Return ONLY JSON with this shape:
{{
  "action": "answer|clarify|unsupported",
  "tables": ["minor_courses"],
  "filters": [{{"field":"batch","operator":"eq|in|contains|gte|lte|gt|lt|neq","value":"2025"}}],
  "select": ["course_code","course_title","credits"],
  "distinct": true,
  "aggregation": {{"function":"none|count|sum|avg|min|max","field":"course_code"}},
  "group_by": [],
  "sort": {{"field":"course_code","direction":"asc"}},
  "limit": 100,
  "relationship": "",
  "clarification_question": "",
  "reason": "",
  "retrieval_query": ""
}}

Rules:
- Map synonyms naturally: subjects/modules/papers/classes -> course records; year/batch/academic year -> batch where appropriate; prerequisite/pre-requisite/pre req -> prerequisite; hours may refer to lecture/tutorial/practical.
- Use the exact database field names above. Never invent fields.
- Preserve every constraint from the question. A single question can contain many filters.
- If a requested fact exists in multiple tables, choose the most specific table. Minor questions -> minor_courses; semester/offering questions -> semester_offerings or semester_courses; programme structure -> structure_courses/degree_requirements; prerequisite relationships -> prerequisites/course_relationships; university policies/SOP/handbook -> policy_documents.
- For counts, use aggregation=count and do not select every row unless needed for explanation.
- For “which courses depend on X”, use course_relationships if possible, otherwise prerequisites with the appropriate relationship.
- For “what is X a prerequisite for”, reverse the prerequisite relationship.
- For comparisons or “year-wise”, group_by the relevant field instead of asking for one year.
- If a minor or structure is versioned by batch/year and the question asks a batch-sensitive fact without specifying the batch, set action=clarify when multiple batches exist.
- If the question asks for something not represented by the available fields, set action=unsupported.
- For a multi-part question, preserve all parts in one plan when possible; otherwise use the first table and select all requested fields.
- Never answer from world knowledge.

USER QUESTION:
{query}
STUDENT PROFILE:
{student_id or 'none'}'''
    obj,err=_call_gemini_json(prompt)
    if not isinstance(obj,dict): return None,err or "No structured plan returned"
    obj.setdefault("action","answer"); obj.setdefault("tables",[]); obj.setdefault("filters",[]); obj.setdefault("select",[])
    obj.setdefault("distinct",True); obj.setdefault("aggregation",{"function":"none","field":""}); obj.setdefault("group_by",[])
    obj.setdefault("sort",{}); obj.setdefault("limit",100); obj.setdefault("relationship",""); obj.setdefault("clarification_question","")
    obj.setdefault("reason",""); obj.setdefault("retrieval_query",query)
    return obj,err


def _generic_apply_filter(df, f):
    field=str(f.get("field","")).strip(); op=str(f.get("operator","eq")).lower(); val=f.get("value","")
    if field not in df.columns: return df
    s=df[field].fillna("").astype(str).str.strip()
    if isinstance(val,list): vals=[str(x).strip().lower() for x in val]
    else: vals=[str(val).strip().lower()]
    sl=s.str.lower()
    if op=="eq": return df[sl.eq(vals[0])]
    if op=="neq": return df[~sl.eq(vals[0])]
    if op=="in": return df[sl.isin(vals)]
    if op=="contains": return df[sl.str.contains(vals[0],regex=False,na=False)]
    # numeric comparisons; nonnumeric rows are excluded rather than guessed
    num=pd.to_numeric(s,errors="coerce"); v=pd.to_numeric(pd.Series([val]),errors="coerce").iloc[0]
    if pd.isna(v): return df
    if op=="gte": return df[num>=v]
    if op=="lte": return df[num<=v]
    if op=="gt": return df[num>v]
    if op=="lt": return df[num<v]
    return df


def _generic_evidence(rows, table):
    if rows is None or len(rows)==0: return []
    ev=[]
    for _,r in rows.head(8).iterrows():
        source=str(r.get("source_sheet","") or "")
        row=str(r.get("source_row","") or "")
        wb=("Minor Courses workbook" if table=="minor_courses" else ("University policy PDF" if table=="policy_documents" else "Semester & Structure workbook"))
        parts=[]
        for c in ["minor","batch","academic_structure","semester","course_code","course_title","credits","prerequisite","component","required_credits","lecture_hours","tutorial_hours","practical_hours","category"]:
            if c in r.index and str(r.get(c,"")) not in {"","nan","None"}: parts.append(f"{c}: {r.get(c)}")
        ev.append({"source":source,"row":row,"workbook":wb,"text":"; ".join(parts)})
    return ev


def execute_generic_plan(plan, rag, original_query):
    # Safely execute the semantic plan over pandas tables; no generated code is executed.
    if not plan: return None
    action=plan.get("action","answer")
    if action=="clarify":
        q=plan.get("clarification_question") or "I need one more detail to answer this accurately. What year, batch, semester, or academic structure should I use?"
        return {"answer":q,"status":"follow_up","evidence":[]}
    if action=="unsupported":
        return {"answer":"I could not map that request to information contained in the supplied academic records. Please ask about courses, minors, prerequisites, offerings, credits, academic structures, requirements, or your synthetic student record.","status":"insufficient","evidence":[]}
    tables=plan.get("tables") or []
    frames={
      "minor_courses":rag["minors"],"semester_courses":rag["semesters"],
      "semester_offerings":rag.get("semester_offerings",rag["semesters"]),
      "structure_courses":rag["structures"],"degree_requirements":rag["requirements"],
      "course_master":rag["courses"],"prerequisites":rag.get("prerequisites",pd.DataFrame()),
      "course_relationships":rag.get("course_relationships",pd.DataFrame()),
      "synthetic_students":rag.get("students",pd.DataFrame()),"student_course_history":rag.get("history",pd.DataFrame()),"policy_documents":rag.get("policy_documents",pd.DataFrame())}
    table=next((t for t in tables if t in frames and not frames[t].empty),None)
    if not table: return None
    df=frames[table].copy()
    for f in plan.get("filters") or []: df=_generic_apply_filter(df,f)
    if df.empty:
        return {"answer":"I could not find any records matching all of those conditions in the supplied academic data.","status":"insufficient","evidence":[]}
    # Grouped aggregation/comparison
    agg=plan.get("aggregation") or {}; fn=str(agg.get("function","none")).lower(); field=agg.get("field") or "course_code"
    groups=plan.get("group_by") or []
    if groups:
        valid=[g for g in groups if g in df.columns]
        if valid:
            rows=[]
            for key,g in df.groupby(valid,dropna=False):
                if not isinstance(key,tuple): key=(key,)
                if fn=="count": val=len(g.drop_duplicates(subset=[field])) if field in g.columns else len(g)
                elif fn in {"sum","avg","min","max"}:
                    nums=pd.to_numeric(g[field],errors="coerce").dropna(); val=None if nums.empty else getattr(nums,fn)()
                else: val=len(g)
                rows.append({**{valid[i]:key[i] for i in range(len(valid))},"value":val})
            out=pd.DataFrame(rows)
            lines=["; ".join(f"{c}: {r[c]}" for c in valid)+f" → **{r['value']}**" for _,r in out.iterrows()]
            return {"answer":"Here is the requested comparison/grouped result based on the available academic records:\n\n"+"\n".join(f"- {x}" for x in lines),"status":"verified","evidence":_generic_evidence(df,table)}
    if fn=="count":
        n=len(df.drop_duplicates(subset=[field])) if field in df.columns else len(df)
        return {"answer":f"The requested count is **{n}** based on the matching records in the supplied academic data.","status":"verified","evidence":_generic_evidence(df,table)}
    if fn in {"sum","avg","min","max"} and field in df.columns:
        nums=pd.to_numeric(df[field],errors="coerce").dropna()
        if not nums.empty:
            val=getattr(nums,fn)()
            return {"answer":f"The {fn} of **{field.replace('_',' ')}** is **{val:g}**, based on the matching supplied records.","status":"verified","evidence":_generic_evidence(df,table)}
    # Selection/list/detail
    select=[x for x in (plan.get("select") or []) if x in df.columns]
    if not select:
        select=[c for c in (["document","page","section","text"] if table=="policy_documents" else ["course_code","course_title","credits","semester","prerequisite"]) if c in df.columns]
    out=df[select].copy()
    if plan.get("distinct",True): out=out.drop_duplicates()
    sort=plan.get("sort") or {}; sf=sort.get("field");
    if sf in out.columns: out=out.sort_values(sf,ascending=str(sort.get("direction","asc")).lower()!="desc")
    try: out=out.head(max(1,min(int(plan.get("limit",100)),100)))
    except Exception: out=out.head(100)
    lines=[]
    for _,r in out.iterrows():
        lines.append(" — ".join(f"{c.replace('_',' ').title()}: {r[c]}" for c in select if str(r[c]) not in {"","nan","None"}))
    return {"answer":f"I found **{len(out)} matching record(s)** in the supplied academic data:\n\n"+"\n".join(f"- {x}" for x in lines),"status":"verified","evidence":_generic_evidence(df,table)}

def deterministic_query_plan(query, minors, semesters, structures):
    """Build a conservative query plan from the user's wording.
    Gemini may enrich this plan, but deterministic parsing remains the fallback/source-of-truth.
    """
    ql=norm(query).lower()
    years=extract_years(ql)
    sem=_semester_number(ql)
    minor=_minor_match(ql,minors)
    operation="answer"
    if any(x in ql for x in ["how many","how much","count","number of","total"]): operation="count"
    elif any(x in ql for x in ["compare","comparison","difference between","year wise","year-wise","by year","across years","across batches"]): operation="compare"
    elif any(x in ql for x in ["list","show","which","what are","give me","tell me the courses","subjects"]): operation="list"
    elif any(x in ql for x in ["prerequisite","prereq","pre req","before taking","before i take"]): operation="prerequisite"
    elif any(x in ql for x in ["credit","credits"]): operation="credits"
    filters={}
    cm=re.search(r"\b(\d+(?:\.\d+)?)\s*credits?\b",ql)
    if cm: filters["credits"]=cm.group(1)
    if any(x in ql for x in ["no prerequisite","without prerequisite","nil prerequisite","no prereq","without prereq"]): filters["prerequisite"]="none"
    elif any(x in ql for x in ["with prerequisite","has prerequisite","have prerequisite"]): filters["prerequisite"]="has"
    return {"intent": "minor" if minor else "unknown", "minor": minor, "years": years, "semester": sem, "operation": operation, "filters": filters}

def database_query_answer(query, minors, semesters, structures, courses, plan=None):
    """Answer questions whose facts can be obtained directly from the two Excel workbooks.
    This is deliberately deterministic; the LLM only verbalizes grounded evidence later.
    Returns None when the query needs course/entity resolution or student logic.
    """
    ql=query.lower()
    plan=plan or deterministic_query_plan(query,minors,semesters,structures)
    # ----- minor database -----
    minor=plan.get("minor") or _minor_match(ql,minors)
    minor_context = any(x in ql for x in ["minor", "minor courses", "minor course", "specialization in", "minor subjects", "subjects for", "courses for"]) or minor is not None
    if minor and minor_context:
        rows=minors[minors.minor.astype(str).str.lower().eq(minor.lower())].copy()
        years=plan.get("years") or _minor_year_intent(ql)[0]
        all_years=(_minor_year_intent(ql)[1] or plan.get("operation")=="compare")
        sem=plan.get("semester") or _semester_number(ql)

        # Academic batch/year is a first-class filter. Never silently merge different
        # batches when the user asks for a count, list, prerequisite, credit, semester,
        # hours, or any other minor-level fact.
        available_batches=sorted({str(x).strip() for x in rows.batch.dropna().tolist() if str(x).strip() and str(x).strip().lower() not in {"nan","none","nil"}}) if "batch" in rows.columns else []
        if years and "batch" in rows.columns:
            rows=rows[rows.batch.astype(str).isin(years)]
            if rows.empty:
                return {"answer":f"I could not find {minor} Minor records for {', '.join(years)} in the supplied Excel workbook. Available batches are: **{', '.join(available_batches)}**.","status":"follow_up","evidence":[]}
        elif len(available_batches)>1 and not all_years:
            # If the user asks a batch-sensitive minor question without naming a year,
            # ask before answering. This applies to count/list/details/credits/prereqs,
            # not just the literal phrase 'how many courses'.
            return {"answer":f"I found **{minor} Minor** records for multiple academic batches: **{', '.join(available_batches)}**. Which year/batch should I use? The course set can differ by batch.","status":"follow_up","evidence":[]}

        # Optional semester filter after batch selection.
        if sem and "semester" in rows.columns:
            rows=rows[rows.semester.astype(str).str.upper().str.replace(" ","").eq(sem)]

        # Cross-year comparison is explicit and safe to answer. Respect whether the
        # user asks for counts or the actual course lists.
        if all_years:
            q_all_list=any(x in ql for x in ["list", "show", "give me all", "what are", "which courses", "subjects"])
            if q_all_list and not any(x in ql for x in ["how many", "count", "number of"]):
                blocks=[]
                for y in available_batches:
                    rr=minors[(minors.minor.astype(str).str.lower()==minor.lower()) & (minors.batch.astype(str)==y)].copy()
                    if sem: rr=rr[rr.semester.astype(str).str.upper().str.replace(" ","").eq(sem)]
                    rr=rr.drop_duplicates(subset=[c for c in ["course_code","course_title"] if c in rr.columns])
                    items=_format_rows(rr,[("course_code","Code"),("course_title","Course"),("credits","Credits"),("semester","Semester"),("prerequisite","Prerequisite")])
                    blocks.append(f"**{y}** — {len(rr)} course(s)\n" + "\n".join(f"- {x}" for x in items))
                return {"answer":f"Here are the **{minor} Minor** courses by academic batch:\n\n"+"\n\n".join(blocks),"status":"verified","evidence":_source_evidence(minors[minors.minor.astype(str).str.lower().eq(minor.lower())])}
            summary=[]
            for y in available_batches:
                rr=minors[(minors.minor.astype(str).str.lower()==minor.lower()) & (minors.batch.astype(str)==y)]
                if sem: rr=rr[rr.semester.astype(str).str.upper().str.replace(" ","").eq(sem)]
                rr=rr.drop_duplicates(subset=[c for c in ["course_code","course_title"] if c in rr.columns])
                summary.append((y,len(rr)))
            detail="\n".join(f"- **{y}:** {n} course(s)" for y,n in summary)
            return {"answer":f"For **{minor} Minor**, the supplied workbook records the following batch-wise counts{(' for '+sem) if sem else ''}:\n\n{detail}","status":"verified","evidence":_source_evidence(minors[minors.minor.astype(str).str.lower().eq(minor.lower())])}

        rows=rows.drop_duplicates(subset=[c for c in ["course_code","course_title"] if c in rows.columns])
        if len(rows)==0:
            return {"answer":f"No matching {minor} Minor course records were found for the specified filters in the supplied Excel data.","status":"insufficient","evidence":[]}

        q_has_count=any(x in ql for x in ["how many","how much","count","number of","total number","total courses","total subjects"])
        q_has_list=_wants_list(ql) or any(x in ql for x in ["list","show","which","what are","give me","tell me the courses","subjects in"])
        q_has_prereq="prereq" in ql or "pre req" in ql or "pre-requisite" in ql
        q_has_credit="credit" in ql or "credits" in ql
        q_has_sem="semester" in ql or "which sem" in ql or "offered when" in ql
        q_has_hours=any(x in ql for x in ["lecture hours","tutorial hours","practical hours","l t p","ltp","hours"])

        # Filter expressions that users commonly combine with minor questions.
        credit_match=re.search(r"\b(\d+(?:\.\d+)?)\s*credits?\b",ql)
        if credit_match:
            val=credit_match.group(1)
            rows=rows[rows.credits.astype(str).str.strip().eq(val)]
        if any(x in ql for x in ["no prerequisite","without prerequisite","nil prerequisite","no prereq","without prereq"]):
            rows=rows[rows.prerequisite.map(lambda x: not norm(x) or norm(x).upper() in {"NIL","NONE","NA","N/A","-"})]
        if any(x in ql for x in ["with prerequisite","has prerequisite","have prerequisite"]):
            rows=rows[rows.prerequisite.map(lambda x: bool(norm(x)) and norm(x).upper() not in {"NIL","NONE","NA","N/A","-"})]

        # Credit-value questions should not dump every course unless the user asks
        # which courses have a particular credit value.
        if q_has_credit and not credit_match and not q_has_list and not q_has_count:
            vals=sorted({norm(x) for x in rows.credits if norm(x)})
            return {"answer":f"The recorded credit values for **{minor} Minor**{(' (batch '+years[0]+')') if years else ''} are **{', '.join(vals) if vals else 'not recorded'}** across the matching courses.","status":"verified","evidence":_source_evidence(rows)}

        # Count questions should answer with a clean number and the selected context.
        if q_has_count and not q_has_list:
            scope=[]
            if years: scope.append(f"batch {years[0]}")
            if sem: scope.append(f"{sem.replace('S','Semester ')}")
            if credit_match: scope.append(f"{credit_match.group(1)}-credit courses")
            if "no prerequisite" in ql or "without prerequisite" in ql: scope.append("courses with no prerequisite recorded")
            scope_text=(" for " + ", ".join(scope)) if scope else ""
            return {"answer":f"There are **{len(rows)} course(s)** recorded for **{minor} Minor**{scope_text}, based on the available academic sources.","status":"verified","evidence":_source_evidence(rows)}

        if q_has_list or q_has_prereq or q_has_credit or q_has_sem or q_has_hours or "minor" in ql:
            if q_has_hours:
                fields=[("course_code","Code"),("course_title","Course"),("lecture_hours","Lecture hours"),("tutorial_hours","Tutorial hours"),("practical_hours","Practical hours"),("credits","Credits"),("semester","Semester"),("prerequisite","Prerequisite")]
            elif q_has_prereq:
                fields=[("course_code","Code"),("course_title","Course"),("prerequisite","Prerequisite"),("credits","Credits"),("semester","Semester")]
            elif q_has_credit:
                fields=[("course_code","Code"),("course_title","Course"),("credits","Credits"),("semester","Semester"),("prerequisite","Prerequisite")]
            else:
                fields=[("course_code","Code"),("course_title","Course"),("credits","Credits"),("semester","Semester"),("prerequisite","Prerequisite")]
            items=_format_rows(rows,fields)
            heading=f"The supplied records for **{minor} Minor**"
            if years: heading+=f" (batch {years[0]})"
            if sem: heading+=f" in {sem.replace('S','Semester ')}"
            return {"answer":heading+f" contain **{len(rows)} course(s)**:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":_source_evidence(rows)}

    # ----- semester-spread database -----
    sem=_semester_number(ql)
    if sem and (_wants_list(ql) or any(x in ql for x in ["semester", "offered", "offering"])):
        rows=semesters[semesters.semester.astype(str).str.upper().eq(sem)].copy()
        # If a structure/year is named, constrain to that sheet.
        stname=_structure_match(ql,semesters.rename(columns={"source_sheet":"source_sheet"})) if False else None
        years=re.findall(r"20(?:22|23|24|25|26)",ql)
        if years:
            year=years[0]
            yrrows=rows[rows.source_sheet.astype(str).str.contains(year,case=False,na=False)]
            if len(yrrows): rows=yrrows
        rows=rows.drop_duplicates(subset=["course_code","course_title"])
        if len(rows):
            fields=[("course_code","Code"),("course_title","Course"),("credits","Credits"),("category","Category"),("basket","Basket"),("prerequisite","Prerequisite")]
            items=_format_rows(rows,fields)
            return {"answer":f"The supplied semester-spread workbook lists {len(rows)} distinct course records for **{sem}**:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":_source_evidence(rows)}

    # ----- structure database -----
    stname=_structure_match(ql,structures)
    if any(x in ql for x in ["structure", "university core", "program core", "programme core", "foundation", "elective", "graduation credit", "credit requirement", "degree credit"]):
        if not stname:
            # Questions explicitly asking for a cross-structure comparison can be answered without guessing.
            if "different structures" in ql or "across structures" in ql or "all structures" in ql:
                vals=[]
                for sh in sorted(structures.source_sheet.dropna().unique()):
                    rr=structures[structures.source_sheet.eq(sh)]
                    vals.append(f"{sh}: {len(rr)} course entries")
                return {"answer":"The supplied workbook contains multiple academic structures:\n\n"+"\n".join(f"- {x}" for x in vals)+"\n\nPlease specify the structure/year for a course-level or credit-requirement answer.","status":"follow_up","evidence":[]}
            return {"answer":"Please specify the applicable academic structure/year (for example, Struct_2024, Struct_2025, or Struct_2026_DS). The supplied workbook contains multiple structures, so I will not assume one.","status":"follow_up","evidence":[]}
        rows=structures[structures.source_sheet.eq(stname)].copy()
        # category filter
        cat=None
        for c in ["university core","program core","programme core","foundation","elective","honors","specialization"]:
            if c in ql: cat=c; break
        if cat:
            if cat in {"program core","programme core"}: rows=rows[rows.category.astype(str).str.contains("Program Core",case=False,na=False)]
            elif cat=="university core": rows=rows[rows.category.astype(str).str.contains("University Core",case=False,na=False)]
            elif cat=="foundation": rows=rows[rows.category.astype(str).str.contains("Foundation",case=False,na=False)]
            elif cat=="elective": rows=rows[rows.category.astype(str).str.contains("Elective",case=False,na=False)]
            elif cat=="honors": rows=rows[rows.category.astype(str).str.contains("Honors|Hons",case=False,na=False)]
            elif cat=="specialization": rows=rows[rows.category.astype(str).str.contains("Specialization|Sp Track|Applied",case=False,na=False)]
        rows=rows.drop_duplicates(subset=["course_title","category"])
        if len(rows) and (_wants_list(ql) or cat):
            items=_format_rows(rows,[("course_title","Course"),("credits","Credits"),("category","Category")])
            return {"answer":f"For **{stname}**, the supplied structure workbook records {len(rows)} matching entries:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":_source_evidence(rows)}

    # ----- generic field queries -----
    # Count / filtering by credits across the course records.
    m=re.search(r"\b(\d+(?:\.\d+)?)\s*credits?\b",ql)
    if m and _wants_list(ql):
        val=m.group(1)
        pool=pd.concat([courses.assign(source_group="course"),structures.assign(source_group="structure")],ignore_index=True,sort=False)
        rows=pool[pool.credits.astype(str).str.strip().eq(val)]
        rows=rows.drop_duplicates(subset=[c for c in ["course_code","course_title","category"] if c in rows.columns])
        if len(rows):
            items=_format_rows(rows,[("course_code","Code"),("course_title","Course"),("credits","Credits"),("category","Category")])
            return {"answer":f"I found {len(rows)} matching records with **{val} credits** in the supplied academic data:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":_source_evidence(rows)}

    # No-prerequisite / prerequisite existence query.
    if _wants_list(ql) and "prerequisite" in ql and any(x in ql for x in ["no prerequisite","without prerequisite","nil prerequisite","prerequisite is nil"]):
        rows=courses[courses.prerequisite.map(lambda x: not norm(x) or norm(x).upper() in {"NIL","NONE","NA","N/A","-"})].drop_duplicates(subset=["course_code","course_title"])
        items=_format_rows(rows,[("course_code","Code"),("course_title","Course"),("credits","Credits")])
        return {"answer":f"The supplied course records contain {len(rows)} courses with no prerequisite recorded:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":_source_evidence(rows)}

    return None

def prerequisite_tokens(pre):
    p=norm(pre)
    if not p or p.upper() in {"NIL","NONE","NA","N/A","-"}: return []
    return [norm_code(x) for x in re.findall(r"[A-Za-z]{2,8}\d{3}",p)]

def student_eligibility(student_id, course, history):
    req=prerequisite_tokens(course.get("prerequisite",""))
    if not req: return {"status":"eligible_on_prerequisite_check","required":[],"missing":[],"failed":[]}
    h=history[history.student_id.eq(student_id)]
    passed=set(norm_code(x) for x in h.loc[h.status.astype(str).str.lower().eq("passed"),"course_code"])
    failed=set(norm_code(x) for x in h.loc[~h.status.astype(str).str.lower().eq("passed"),"course_code"])
    missing=[x for x in req if x not in passed and x not in failed]
    failed_req=[x for x in req if x in failed]
    status="not_eligible_based_on_record" if failed_req else ("insufficient_student_record" if missing else "eligible_on_prerequisite_check")
    return {"status":status,"required":req,"missing":missing,"failed":failed_req}

def material_conflicts(course_code, semesters):
    """Only flag a true within-version disagreement. Different academic years are versioned records, not automatic conflicts."""
    rows=semesters[semesters.course_code.astype(str).str.upper().eq(course_code.upper())]
    conflicts=[]
    for sheet,grp in rows.groupby("source_sheet"):
        vals=[]
        for x in grp.prerequisite.astype(str):
            v=norm(x)
            if v not in vals: vals.append(v)
        if len(vals)>1:
            conflicts.append({"source_sheet":sheet,"prerequisites":vals,"sources":[f"{r.source_sheet} row {r.source_row}" for _,r in grp.iterrows()]})
    return conflicts

def _normalise_chat_text(query):
    q = re.sub(r"[^a-z0-9\s\']", " ", norm(query).lower())
    return re.sub(r"\s+", " ", q).strip()

def _is_small_talk(query):
    """Handle only standalone social messages. Addressing AIRA is allowed to reach Gemini."""
    q = _normalise_chat_text(query)
    return q in {"hi","hello","hey","hi there","hello there","hey there","good morning","good afternoon","good evening","thanks","thank you","thanks a lot","bye","goodbye"}

def _is_aira_greeting(query):
    """A greeting explicitly addressed to AIRA should be processed by Gemini, not hard-coded."""
    q = _normalise_chat_text(query)
    return bool(re.fullmatch(r"(?:hi|hello|hey)(?:\s+there)?\s+aira", q) or
                re.fullmatch(r"(?:hi|hello|hey)\s*,?\s+aira", q))

def policy_document_fallback(query, rag):
    """Retrieve policy/SOP evidence when the semantic planner cannot map the question.
    The answer is deliberately evidence-first; Gemini later verbalizes only these excerpts."""
    docs=rag.get("policy_documents",pd.DataFrame())
    if docs.empty: return None
    qwords=[w for w in _query_words(query) if len(w)>2]
    if not qwords: return None
    def score(row):
        text=str(row.get("text","" )).lower(); section=str(row.get("section","" )).lower();
        hits=sum(1 for w in qwords if w in text or w in section)
        phrase=sum(1 for a,b in [("attendance","attendance"),("add drop","course add/drop"),("registration","registration"),("audit","audit course"),("summer","summer term"),("digii","digii"),("grievance","grievance"),("scholarship","scholarship"),("parking","parking"),("progression","progression"),("transfer","transfer of credits"),("degree","award of degree")] if a in query.lower() and b in text)
        # Contents pages are navigation aids, not substantive policy evidence.
        penalty=4 if str(row.get("document","")).startswith("Student Handbook") and int(row.get("page",0) or 0)<=5 else 0
        section_bonus=3 if section and any(w in section for w in qwords) else 0
        ql=query.lower()
        if "attendance requirement" in ql and ("seventy five percent" in text or "attendance requirements" in section): phrase += 10
        if ("one scholarship" in ql or "more than one scholarship" in ql) and "only one scholarship" in text.lower(): phrase += 10
        if ("speed limit" in ql or "vehicle" in ql) and "20 kmph" in text.lower(): phrase += 10
        if "award of a degree" in ql and "award of the degree" in text.lower(): phrase += 10
        if "add/drop" in ql or "add drop" in ql or ("add" in ql and "drop" in ql):
            if "course add/drop" in text.lower(): phrase += 10
        if "audit" in ql and ("audit course registration" in text.lower() or "audit a course" in text.lower()): phrase += 10
        return hits+2*phrase+section_bonus-penalty
    tmp=docs.copy(); tmp["_score"]=tmp.apply(score,axis=1); tmp=tmp[tmp._score>0].sort_values("_score",ascending=False).head(5)
    if tmp.empty: return None
    ev=[]
    for _,r in tmp.iterrows():
        ev.append({"source":r.get("document",""),"row":r.get("page",""),"workbook":r.get("source_file","University policy PDF"),"text":str(r.get("text",""))[:1200],"page":r.get("page","")})
    excerpts=[]
    for e in ev[:4]:
        excerpts.append(f"- **{e['source']} — page {e['page']}**: {e['text'][:700]}")
    return {"answer":"The supplied Student Handbook/SOP contains the following relevant records:\n\n"+"\n".join(excerpts),"status":"verified","evidence":ev}

def direct_answer(query, student_id, rag, plan=None):
    minors, semesters, structures, requirements, courses = rag["minors"],rag["semesters"],rag["structures"],rag["requirements"],rag["courses"]
    q=norm(query); ql=q.lower(); evidence=[]

    # Standalone greetings can stay instant. A greeting explicitly addressed to
    # AIRA (e.g. "Hi AIRA") is intentionally NOT short-circuited: Gemini gets to
    # understand and generate the conversational response.
    if _is_aira_greeting(q):
        return {
            "answer":"The user is greeting AIRA directly. Respond naturally, warmly, and invite the user to ask an academic question. Do not invent academic facts.",
            "status":"conversation",
            "evidence":[]
        }
    if _is_small_talk(q):
        if ql in {"bye", "goodbye"}:
            return {"answer":"Goodbye!", "status":"greeting", "evidence":[]}
        if ql in {"thanks", "thank you", "thanks a lot"}:
            return {"answer":"You're welcome!", "status":"greeting", "evidence":[]}
        return {"answer":"Hi! How can I help with your academic records?", "status":"greeting", "evidence":[]}
    # University policy/SOP questions are routed to the PDF evidence store.
    policy_terms=["attendance","registration","add/drop","add drop","add or drop","audit course","audit a course","summer term","digii","grievance","scholarship","parking","helmet","vehicle","speed limit","progression","transfer of credits","award of degree","award of a degree","bonafide","fee","late registration"]
    if any(t in ql for t in policy_terms) and not any(x in ql for x in ["prerequisite","prereq"]):
        pr=policy_document_fallback(q,rag)
        if pr is not None: return pr

    # Handle generic prerequisite filters/policy-style source-bound questions
    # before relationship parsing; these do not name a target course.
    if "failed" in ql and "prerequisite" in ql:
        return {"answer":"If a required prerequisite is recorded as failed, the supplied student-history check treats that prerequisite as not satisfied. The supplied sources do not establish an override or exception process.","status":"verified","evidence":[]}
    if _wants_list(ql) and "prerequisite" in ql and any(x in ql for x in ["no prerequisite","without prerequisite","nil prerequisite","prerequisite is nil"]):
        rows=courses[courses.prerequisite.map(lambda x: not norm(x) or norm(x).upper() in {"NIL","NONE","NA","N/A","-"})].drop_duplicates(subset=["course_code","course_title"])
        items=_format_rows(rows,[("course_code","Code"),("course_title","Course"),("credits","Credits")])
        return {"answer":f"The supplied course records contain {len(rows)} courses with no prerequisite recorded:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":_source_evidence(rows)}
    prereq_route=prerequisite_relationship_answer(q,courses)
    if prereq_route is not None:
        return prereq_route
    # First route questions that can be answered directly from workbook tables.
    plan=plan or deterministic_query_plan(q,minors,semesters,structures)
    routed=database_query_answer(q,minors,semesters,structures,courses,plan)
    if routed is not None:
        return routed
    if not student_id:
        sid_hit=re.search(r"\bSYN\d{3}\b",q.upper())
        if sid_hit: student_id=sid_hit.group(0)

    # 1) Broad source-database questions first.
    minor_name=_minor_match(ql,minors)
    if minor_name and (_is_list_query(ql) or "minor" in ql) and not any(x in ql for x in ["prereq","credit of","when"]):
        rows=minors[minors.minor.astype(str).str.lower().eq(minor_name.lower())].copy()
        rows=rows.drop_duplicates(subset=["course_code","course_title"]) if "course_code" in rows else rows.drop_duplicates()
        if len(rows):
            items=[f"{display_course_code(r.course_code)} — {norm(r.course_title)} ({norm(r.credits)} credits; Semester {norm(r.semester)})" for _,r in rows.iterrows()]
            evidence=_source_evidence(rows)
            return {"answer":f"The supplied **{minor_name}** minor data contains {len(items)} distinct course records:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":evidence}

    semno=_semester_number(ql)
    if semno and _is_list_query(ql):
        rows=semesters[semesters.semester.astype(str).str.upper().eq(semno)].drop_duplicates(subset=["course_code","course_title"])
        if len(rows):
            items=[f"{display_course_code(r.course_code)} — {norm(r.course_title)} ({norm(r.credits)} credits)" for _,r in rows.iterrows()]
            evidence=_source_evidence(rows)
            return {"answer":f"The supplied semester-spread workbook lists {len(items)} distinct courses in **{semno}**:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":evidence}

    struct=_structure_match(ql,structures)
    if struct and any(x in ql for x in ["graduation credit","credit requirement","credits required","degree credits","how many credits"]):
        req_rows=requirements[requirements.source_sheet.eq(struct)].copy()
        if len(req_rows):
            summary=[f"{norm(r.basket)}: {norm(r.credits)} credits" for _,r in req_rows.iterrows()]
            ev=[{"source":r.source_sheet,"row":r.source_row,"text":f"{norm(r.basket)}: {norm(r.credits)} credits","workbook":"Semester & Structure workbook"} for _,r in req_rows.iterrows()]
            return {"answer":f"For **{struct}**, the supplied structure workbook records these credit requirements:\n\n"+"\n".join(f"- {x}" for x in summary),"status":"verified","evidence":ev}

    if struct and (_is_list_query(ql) or "structure" in ql or "program core" in ql or "programme core" in ql or "university core" in ql or "foundation" in ql or "elective" in ql):
        rows=structures[structures.source_sheet.eq(struct)].copy()
        if "program core" in ql or "programme core" in ql: rows=rows[rows.category.astype(str).str.contains("Program Core",case=False,na=False)]
        elif "university core" in ql: rows=rows[rows.category.astype(str).str.contains("University Core",case=False,na=False)]
        elif "foundation" in ql: rows=rows[rows.category.astype(str).str.contains("Foundation",case=False,na=False)]
        elif "elective" in ql: rows=rows[rows.category.astype(str).str.contains("Elective",case=False,na=False)]
        rows=rows.drop_duplicates(subset=["course_title","category"])
        if len(rows):
            items=[f"{norm(r.course_title)} ({norm(r.credits)} credits)" for _,r in rows.iterrows()]
            evidence=_source_evidence(rows)
            return {"answer":f"For **{struct}**, the supplied structure workbook records {len(items)} matching course entries:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":evidence}

    # Broad category questions without a structure/year must be clarified because
    # the two workbooks contain multiple academic structures.
    if any(x in ql for x in ["university core","foundation category","foundation courses","program core","programme core","electives"]):
        if not struct:
            return {"answer":"I need the applicable academic structure/year before I can give a reliable course list for that category because the supplied workbook contains multiple structures. Please provide the year/structure (for example, 2024, 2025, or 2026).","status":"follow_up","evidence":[]}

    if "minor courses" in ql and not minor_name:
        return {"answer":"Please provide the minor name (for example, Finance, Marketing, Economics, Psychology, Design, Law, or Start-up) so I can retrieve the relevant course records.","status":"follow_up","evidence":[]}

    if "offering" in ql and "minor" in ql and not minor_name:
        return {"answer":"Please provide the minor name so I can identify its course-offering records.","status":"follow_up","evidence":[]}

    if "guarantee" in ql and "offering" in ql:
        return {"answer":"No. A semester listing in the supplied workbook is evidence that the course appears in that semester-spread record; it does not establish a guarantee of future registration availability.","status":"verified","evidence":[]}

    if "failed" in ql and "prerequisite" in ql:
        return {"answer":"If a required prerequisite is recorded as failed, the supplied student-history check treats that prerequisite as not satisfied. The supplied sources do not establish an override or exception process.","status":"verified","evidence":[]}

    if any(x in ql for x in ["source records disagree", "records disagree", "conflicting source", "source conflict", "two source records"]):
        return {"answer":"When source records materially disagree, AIRA does not silently choose one. It identifies the conflicting records, preserves their sheet/row evidence, and asks the user to verify the applicable official academic record.","status":"verified","evidence":[]}

    if "credit requirements for different structures" in ql or "different structures" in ql and "credit" in ql:
        wb2=openpyxl.load_workbook(STRUCT_FILE,data_only=True); allsum=[]
        for ws2 in wb2.worksheets[5:]:
            vals2=list(ws2.iter_rows(values_only=True))
            for rr in vals2:
                if len(rr)>=15 and norm(rr[13]) and norm(rr[14]) and norm(rr[13]).lower() not in {"\"baskets\"","baskets"}:
                    allsum.append(f"{ws2.title}: {norm(rr[13])} — {norm(rr[14])} credits")
        return {"answer":"Credit requirements recorded across the supplied structures:\n\n"+"\n".join(f"- {x}" for x in allsum),"status":"verified","evidence":[{"source":"structure sheets","row":"","text":x} for x in allsum]}

    # 2) Course-name/code resolution. Course name is a first-class input.
    kind, course=find_course(q,courses)
    if kind=="ambiguous":
        rows=course.drop_duplicates(subset=["course_code","course_title"])
        choices=", ".join(sorted(f"{display_course_code(r.course_code)} — {norm(r.course_title)}" for _,r in rows.iterrows()))
        return {"answer":f"I found multiple possible matches: **{choices}**. Please tell me which course you mean. I will not guess.","status":"follow_up","evidence":_source_evidence(rows)}
    if kind=="none":
        if any(x in ql for x in ["can i", "can ", "eligible", "register", "enroll", "enrol", "am i allowed"]):
            return {"answer":"To check eligibility, please provide/select the course you mean. If you are asking about your own eligibility, I also need the relevant student profile or synthetic student ID.","status":"follow_up","evidence":[]}
        if "credit" in ql and any(x in ql for x in ["degree","graduat","required","need"]):
            cats=structures.groupby(["source_sheet","category"],dropna=False).size().reset_index()
            return {"answer":"I need the applicable academic structure/batch to determine a graduation-credit requirement because the supplied workbook contains multiple structure years. Please provide the structure/year (for example, 2024, 2025, or 2026).","status":"follow_up","evidence":_source_evidence(cats)}
        if any(x in ql for x in ["can i","eligible","register","enroll","enrol"]):
            return {"answer":"I could not identify the course in the supplied academic data. Please provide the course name or code exactly as it appears in the university records.","status":"follow_up","evidence":[]}
        # A concrete-looking course code/name that cannot be resolved is a clarification case,
        # not proof that the entire academic question is unsupported.
        code_matches=re.findall(r"\b[A-Z]{2,8}\d{3}\b", q.upper())
        unknown_codes=[c for c in code_matches if not len(courses[courses.course_code.eq(norm_code(c))])]
        if unknown_codes or "course called" in ql or "course named" in ql or ("course code" in ql and not code_matches):
            return {"answer":"I could not match that course to the supplied records. Please provide the exact course name or course code as it appears in the university records.","status":"follow_up","evidence":[]}
        return {"answer":"I could not identify the requested information in the supplied university Excel sources. Please provide a more specific course name, course code, minor, semester, structure/year, or other detail.","status":"insufficient","evidence":[]}

    code=norm_code(course.course_code); title=norm(course.course_title)
    fact=_course_fact_answer(code,title,courses,semesters,ql)
    conflicts=material_conflicts(code,semesters) if code else []
    if conflicts:
        base_ev=_source_evidence(_course_rows(courses,code))
        return {"answer":f"The supplied semester-spread data contains differing prerequisite records for **{display_course_code(code)} — {title}**. I will not silently choose between them. The source records should be verified against the current official academic authority.","status":"conflict","evidence":base_ev}
    if fact:
        # Student-specific wording still requires a profile.
        if any(x in ql for x in ["can i","can ","eligible","register","enroll","enrol","am i allowed"]):
            pass
        else: return fact

    # 3) Student eligibility.
    if any(x in ql for x in ["can i","can we","can ","eligible","register","enroll","enrol","am i allowed"]):
        if not student_id:
            return {"answer":f"I found **{display_course_code(code)} — {title}**. To check your eligibility, please select a synthetic student profile so I can compare its recorded history with the source prerequisite.","status":"follow_up","evidence":_source_evidence(_course_rows(courses,code))}
        _,history=synthetic_students(); result=student_eligibility(student_id,course.to_dict(),history)
        ev=_source_evidence(_course_rows(courses,code))
        if result["status"]=="eligible_on_prerequisite_check":
            ans=f"For **{student_id}**, the recorded prerequisite evidence for {display_course_code(code)} is satisfied. This verifies the prerequisite check only; the supplied sources do not establish every possible registration rule."
        elif result["status"]=="not_eligible_based_on_record":
            ans=f"For **{student_id}**, the recorded history does not satisfy the prerequisite check for {display_course_code(code)}: a required prerequisite is recorded as failed. The supplied sources do not establish an override process."
        else:
            ans=f"I do not have enough student-record information to verify eligibility for {display_course_code(code)}. Required prerequisite evidence not demonstrated: {', '.join(result['missing'])}."
        return {"answer":ans,"status":"verified" if result["status"]!="insufficient_student_record" else "insufficient","evidence":ev}

    # 4) General course fact response covers combined questions.
    rows=_course_rows(courses,code)
    ev=_source_evidence(rows)
    credits=sorted(set(norm(x) for x in rows.credits if norm(x)))
    pres_raw=[norm(x) for x in rows.prerequisite if norm(x)]
    pres=[]
    for x in pres_raw:
        if x.upper() in {"NIL","NONE","NA","N/A","-"}: continue
        if x not in pres: pres.append(x)
    if not pres and pres_raw: pres=["Nil/none recorded"]
    return {"answer":f"**{code} — {title}**\n\nCredits: {', '.join(credits) if credits else 'not recorded'}\n\nPrerequisite: {', '.join(pres) if pres else 'Nil/none recorded'}.","status":"verified","evidence":ev}

# ---------- LLM ----------
SYSTEM = """You are AIRA, an academic decision-support assistant.
Use ONLY the supplied university evidence and synthetic student data.
Never invent a course, prerequisite, credit, offering, policy, or registration rule.
If evidence is missing, ambiguous, or conflicting, say so and ask the smallest useful follow-up question.
If structured verification disagrees with an LLM inference, structured verification wins.
Do not claim that an offering record guarantees future availability.
Answer concisely. Do not provide unsupported recommendations."""

def call_gemini(user_prompt):
    key=st.secrets.get("GEMINI_API_KEY",os.getenv("GEMINI_API_KEY",""))
    if not key: return None,"GEMINI_API_KEY is not configured."
    try:
        from google import genai
        client=genai.Client(api_key=key)
        models=[st.secrets.get("GEMINI_MODEL","gemini-2.5-flash"),
                "gemini-2.5-flash-lite"]
        last=None
        for model in dict.fromkeys(models):
            try:
                r=client.models.generate_content(model=model,contents=SYSTEM+"\n\n"+user_prompt)
                txt=getattr(r,"text",None)
                if txt: return txt,None
            except Exception as e: last=e
        return None,f"Gemini request failed: {last}"
    except Exception as e:
        return None,f"Gemini SDK error: {e}"

def llm_layer_answer(layer, query, evidence_text, structured_text):
    if layer=="Basic LLM":
        prompt=f"""Answer the following question as a general language-model baseline.
Do not claim a university-specific fact unless it is explicitly supplied in the question.
QUESTION:
{query}"""
    elif layer=="Structured Prompting":
        prompt=f"""Use a strict academic-advisor format.
QUESTION:
{query}
OUTPUT:
1. Answer or uncertainty
2. What information is missing, if any
3. Do not invent university-specific facts."""
    elif layer=="LLM + RAG":
        prompt=f"""Answer using ONLY these retrieved university records.
<RETRIEVED_EVIDENCE>
{evidence_text}
</RETRIEVED_EVIDENCE>
QUESTION:
{query}"""
    else:
        prompt=f"""Answer using the retrieved university evidence and verified synthetic student context.
<UNIVERSITY_EVIDENCE>
{evidence_text}
</UNIVERSITY_EVIDENCE>
<STUDENT_CONTEXT>
{structured_text}
</STUDENT_CONTEXT>
QUESTION:
{query}"""
    return call_gemini(prompt)

# ---------- USER-FACING UI ----------
minors, semesters, structures, requirements, courses, placeholders = load_authoritative_data()
students, history = synthetic_students()
rag=build_rag()
# Policy and course-alias tables are part of the same authoritative knowledge base.
rag["policy_documents"] = load_policy_documents()
rag["course_aliases"] = load_course_aliases()
# Expose additional normalized tables to the generic semantic executor.
try:
    _cleaned = pd.ExcelFile(DATA / "CLEANED_ACADEMIC_DATABASE.xlsx")
    for _sheet,_key in [("semester_offerings","semester_offerings"),("prerequisites","prerequisites"),("course_relationships","course_relationships"),("synthetic_students","students"),("student_course_history","history")]:
        if _sheet in _cleaned.sheet_names:
            rag[_key]=pd.read_excel(DATA / "CLEANED_ACADEMIC_DATABASE.xlsx",sheet_name=_sheet)
except Exception:
    rag.setdefault("semester_offerings",pd.DataFrame()); rag.setdefault("prerequisites",pd.DataFrame()); rag.setdefault("course_relationships",pd.DataFrame()); rag.setdefault("students",students); rag.setdefault("history",history)

SOURCE_NAMES = {
    "Minor_Courses_for_BTech_Students(3)": "Minor Courses workbook",
    "Semester_Spread_Structures_Sept_2026(3)": "Semester & Structure workbook",
}

def friendly_source(e):
    raw = str(e.get("source", "University Excel source"))
    if str(e.get("workbook","")).endswith(".pdf") or "Handbook" in raw or "SOP" in raw:
        label = str(e.get("workbook", "University policy PDF"))
    else:
        label = str(e.get("workbook", SOURCE_NAMES.get(raw.replace(".xlsx", ""), raw)))
    row = e.get("row", "")
    if row and str(row) not in {"nan", "None"}:
        return f"{label}  ·  {raw}  ·  Row {row}"
    return f"{label}  ·  {raw}"

def render_sources(evidence):
    if not evidence: return
    seen=set(); shown=0
    for e in evidence:
        key=(str(e.get("source")),str(e.get("row")),str(e.get("text")))
        if key in seen: continue
        seen.add(key); shown += 1
        st.markdown(
            f'<div class="source-card"><div class="source-mark">⌁</div><div>'
            f'<div class="source-label">{friendly_source(e)}</div>'
            f'<div class="source-detail">{norm(e.get("text",""))[:360]}</div></div></div>',
            unsafe_allow_html=True,
        )
        if shown >= 6: break

def apply_conversation_context(query, courses):
    """Resolve conversational references such as ‘this course’ from the last resolved course."""
    q=norm(query)
    ql=q.lower()
    if not re.search(r"\b(this|that|the)\s+(course|subject|module|class)\b|\bit\b", ql):
        return q
    last=st.session_state.get("last_course_code")
    if not last: return q
    hit=courses[courses.course_code.eq(last)]
    if hit.empty: return q
    title=norm(hit.iloc[0].course_title)
    replacement=f"{last} — {title}"
    q=re.sub(r"\b(this|that|the)\s+(course|subject|module|class)\b", replacement, q, flags=re.I)
    q=re.sub(r"\bit\b", replacement, q, flags=re.I)
    return q

def format_production_answer(base):
    """Create a polished student-facing fallback without changing verified facts."""
    answer=str(base.get("answer", "")).strip()
    status=base.get("status", "verified")
    if status == "greeting":
        return answer
    if status == "follow_up":
        return f"### I need one clarification\n\n{answer}"
    if status == "insufficient":
        return f"### I need a little more information\n\n{answer}"
    if status == "conflict":
        return f"### I found different source records\n\n{answer}"

    # Natural prerequisite wording.
    m=re.match(r"^To take \*\*(.+?)\s+—\s+(.+?)\*\*, the prerequisite recorded in the supplied academic data is \*\*(.+?)\*\*\.?$",answer,re.I|re.S)
    if m:
        code,title,pre=m.groups()
        return f"### {code} — {title}\n\nThe prerequisite of this course is **{pre}**, based on the available academic sources."

    m=re.match(r"^\*\*(.+?)\s+—\s+(.+?)\*\* has no prerequisite recorded in the supplied academic data\.?$",answer,re.I|re.S)
    if m:
        code,title=m.groups()
        return f"### {code} — {title}\n\nThis course has **no prerequisite recorded** in the available academic sources."

    m=re.match(r"^(.+?)\s+—\s+(.+?): the prerequisite recorded in the supplied Excel data is \*\*(.+?)\*\*\.?$",answer,re.I|re.S)
    if m:
        code,title,pre=m.groups()
        return f"### {code} — {title}\n\nThe prerequisite of this course is **{pre}**, based on the available academic sources."

    m=re.match(r"^(.+?)\s+—\s+(.+?): prerequisite \*\*(.+?)\*\*; credits \*\*(.+?)\*\*\.?$",answer,re.I|re.S)
    if m:
        code,title,pre,credits=m.groups()
        return f"### {code} — {title}\n\nThe prerequisite of this course is **{pre}**.\n\n**Credits:** {credits}\n\n*Based on the available academic sources.*"

    m=re.match(r"^(.+?)\s+—\s+(.+?): credits recorded in the supplied Excel data: \*\*(.+?)\*\*\.?$",answer,re.I|re.S)
    if m:
        code,title,credits=m.groups()
        return f"### {code} — {title}\n\nThis course carries **{credits} credits**, based on the available academic sources."

    return f"### Answer\n\n{answer}\n\n*Based on the available academic sources.*"


def _gemini_client_and_models():
    # Create a Gemini client and discover usable generateContent models.
    key=st.secrets.get("GEMINI_API_KEY",os.getenv("GEMINI_API_KEY",""))
    if not key:
        return None, [], "GEMINI_API_KEY is not configured in Streamlit Secrets."
    try:
        from google import genai
        client=genai.Client(api_key=key)
        configured=st.secrets.get("GEMINI_MODEL",os.getenv("GEMINI_MODEL","gemini-2.5-flash"))
        candidates=[]
        for x in [configured,"gemini-2.5-flash","gemini-2.5-flash-lite","gemini-2.0-flash"]:
            if x and x not in candidates: candidates.append(x)
        # Discover available models when the SDK exposes model listing.
        try:
            listed=client.models.list()
            for m in listed:
                name=str(getattr(m,"name","") or "")
                actions=getattr(m,"supported_actions",None) or []
                if name and "generateContent" in actions and "gemini" in name.lower():
                    short=name.split("/",1)[-1]
                    if short not in candidates: candidates.append(short)
        except Exception:
            pass
        return client,candidates,None
    except Exception as e:
        return None,[],f"Gemini SDK error: {e}"


def _call_gemini_json(prompt):
    client,models,error=_gemini_client_and_models()
    if not client: return None,error
    import json
    last=None
    for model in models:
        try:
            r=client.models.generate_content(model=model,contents=prompt,config={"response_mime_type":"application/json"})
            txt=getattr(r,"text",None)
            if txt:
                return json.loads(txt),None
        except Exception as e:
            last=e
    return None,f"Gemini request failed: {last}"


def _call_gemini_text(prompt):
    client,models,error=_gemini_client_and_models()
    if not client: return None,error
    last=None
    for model in models:
        try:
            r=client.models.generate_content(model=model,contents=prompt)
            txt=getattr(r,"text",None)
            if txt and txt.strip(): return txt.strip(),None
        except Exception as e:
            last=e
    return None,f"Gemini request failed: {last}"


def gemini_understand_query(query, student_id=None):
    # Actual LLM query understanding; facts are never supplied by this layer.
    prompt=f'''You are AIRA, an academic-advisor query understanding engine.
The university source database contains minor courses, semester courses, programme structures,
prerequisites, credit requirements, offerings, and synthetic student records.

Understand the user's question even if it is informal, misspelled, conversational, or contains
multiple requests. Do NOT answer it. Do NOT invent academic facts.

Return ONLY JSON with exactly these fields:
{{
  "intent": "conversation|minor|course_details|prerequisite|dependent_courses|offering|eligibility|credits|graduation_requirement|structure|comparison|policy|general_academic|ambiguous|insufficient",
  "operation": "count|list|detail|filter|compare|answer|check_eligibility|find_prerequisites|find_dependents",
  "minor": "Finance|Marketing|Economics|Psychology|Design|Law|Start-up|",
  "years": ["2022"],
  "semester": "S1|S2|S3|S4|S5|S6|S7|S8|",
  "course_terms": [],
  "credits": "",
  "filters": {{"no_prerequisite": false, "has_prerequisite": false, "lecture_hours": "", "tutorial_hours": "", "practical_hours": ""}},
  "needs_student_profile": false,
  "needs_clarification": false,
  "clarification_reason": "",
  "retrieval_query": "short precise query preserving all entities and filters"
}}

Rules:
- If a minor is named but no year is given and multiple batches exist, set needs_clarification=true.
- If the user asks a cross-year/year-wise question, preserve that in operation=compare and do not ask for one year.
- Distinguish "what courses depend on X" from "what is prerequisite for X".
- Treat "subjects" as courses.
- Extract years even from "batch 2025", "AY 2025", "2025-26".
- Extract semester from "sem 5", "fifth semester", etc.
- Extract numeric credit filters.
- Keep course names as written; do not replace them with guessed codes.
- "Hi AIRA" is conversation, but academic questions inside the same message take priority.

USER QUESTION:
{query}
SELECTED STUDENT PROFILE:
{student_id or 'none'}
'''
    obj,error=_call_gemini_json(prompt)
    if not isinstance(obj,dict):
        return {"intent":"unknown","operation":"answer","minor":"","years":[],"semester":"","course_terms":[],"credits":"","filters":{},"needs_student_profile":False,"needs_clarification":False,"clarification_reason":"","retrieval_query":query},error
    obj.setdefault("intent","general_academic"); obj.setdefault("operation","answer"); obj.setdefault("minor",""); obj.setdefault("years",[])
    obj.setdefault("semester",""); obj.setdefault("course_terms",[]); obj.setdefault("credits",""); obj.setdefault("filters",{})
    obj.setdefault("needs_student_profile",False); obj.setdefault("needs_clarification",False); obj.setdefault("clarification_reason",""); obj.setdefault("retrieval_query",query)
    return obj,error


def validate_llm_response(text, base, rag):
    """Guard the final LLM wording against introducing unsupported course codes.
    Academic facts remain deterministic; if the LLM invents a course code, use the
    deterministic verified wording instead of exposing an ungrounded response.
    """
    if not text or not isinstance(text, str):
        return False, "empty_response"
    # Never allow source placeholders to appear as if they were academic facts.
    upper_text=text.upper().replace("’", "'")
    if any(token in upper_text for token in ["DON'T KNOW", "DONTKNOW", "UNKNOWN", "NOT KNOWN", "NOT AVAILABLE", "TBD"]):
        return False, "source_placeholder_exposed"
    codes = re.findall(r"\b[A-Z]{2,8}\d{3}\b", text.upper())
    if codes:
        known=set(rag["courses"]["course_code"].astype(str).str.upper())
        unknown=[c for c in codes if c not in known]
        if unknown:
            return False, "unknown_course_code:" + ",".join(sorted(set(unknown)))
    # A verified count should retain the count produced by the source-backed executor.
    if base.get("status") == "verified" and re.search(r"\b(count|number|total)\b", str(base.get("answer","")), re.I):
        base_nums=re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])", str(base.get("answer","")))
        # Do not require every number to be copied; require the first explicit count
        # when the verified answer contains a single salient integer.
        if len(base_nums)==1 and base_nums[0] not in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])", text):
            return False, "verified_count_missing"
    return True, ""

def _deterministic_plain_language(base):
    status=base.get("status","verified")
    answer=str(base.get("answer","")).strip()
    if status=="greeting":
        return {"headline":"Hi!", "direct_answer":answer or "How can I help you with the academic records?", "simple_explanation":"You can ask about courses, minors, prerequisites, credits, semester offerings, academic structures, or your synthetic student record.", "details":[], "note":""}
    if status=="follow_up":
        return {"headline":"I need one detail", "direct_answer":answer, "simple_explanation":"I am asking because the available records allow more than one interpretation. Choosing one without your confirmation could give you the wrong academic answer.", "details":[], "note":"AIRA does not guess when an academic detail is ambiguous."}
    if status=="insufficient":
        return {"headline":"I do not have enough information", "direct_answer":answer, "simple_explanation":"The supplied academic records do not contain enough verified information to answer this reliably.", "details":[], "note":"AIRA does not fill missing information with outside assumptions."}
    if status=="conflict":
        return {"headline":"The available records differ", "direct_answer":answer, "simple_explanation":"Different supplied records give different academic information, so AIRA shows the difference instead of choosing one without evidence.", "details":[], "note":"Specify the applicable academic batch or structure if one applies."}
    count_match=re.search(r"\bthere are\s+\*\*(\d+)\s+course", answer, re.I)
    if count_match:
        return {"headline":"Answer", "direct_answer":answer, "simple_explanation":f"In simple terms, the available records show {count_match.group(1)} course(s) matching the scope stated in the answer.", "details":[], "note":"The count is based only on the supplied academic records."}
    if "prerequisite" in answer.lower():
        return {"headline":"Answer", "direct_answer":answer, "simple_explanation":"A prerequisite is a course or requirement that the supplied records say should be completed before taking the course in question.", "details":[], "note":"This explanation does not add any prerequisite beyond the verified result."}
    if "course(s)" in answer.lower() or "matching record" in answer.lower():
        return {"headline":"Answer", "direct_answer":answer, "simple_explanation":"The result contains only records that match the conditions identified from your question.", "details":[], "note":"The result is based only on the supplied academic records."}
    return {"headline":"Answer", "direct_answer":answer, "simple_explanation":"The answer above is based on the supplied academic records. No additional academic facts have been added.", "details":[], "note":""}


def _render_structured_response(obj):
    if not isinstance(obj,dict): return ""
    headline=norm(obj.get("headline","Answer")) or "Answer"
    direct=str(obj.get("direct_answer","")).strip()
    simple=str(obj.get("simple_explanation","")).strip()
    details=obj.get("details") or []
    note=str(obj.get("note","")).strip()
    parts=[f"### {headline}"]
    if direct: parts.append(direct)
    if simple and simple.lower()!=direct.lower(): parts.extend(["**In simple terms**", simple])
    clean=[]
    if isinstance(details,list):
        for d in details:
            d=str(d).strip()
            if d and d not in clean: clean.append(d)
    if clean: parts.extend(["**Details**", "\n".join(f"- {d}" for d in clean[:12])])
    if note: parts.extend(["**Note**", note])
    return "\n\n".join(parts)


def gemini_structured_response(query, base, student_id=None):
    status=base.get("status","verified")
    evidence=base.get("evidence",[])
    evidence_text="\n".join(f"- {e.get('text','')} | workbook={e.get('workbook','')} | sheet={e.get('source','')} | row={e.get('row','')}" for e in evidence)
    schema={"headline":"short heading","direct_answer":"direct answer in simple language","simple_explanation":"one short explanation for a new user unfamiliar with university terminology","details":["only verified details from the result"],"note":"scope or uncertainty note if needed"}
    prompt=f'''You are AIRA, a clear and careful university academic advisor.

The VERIFIED_RESULT below has already been computed from the supplied academic Excel records. Your job is ONLY to explain that verified result to the student.

Return ONLY valid JSON matching this schema:
{json.dumps(schema, ensure_ascii=False)}

NON-NEGOTIABLE RULES:
1. VERIFIED_RESULT is the factual source of truth. Never add an academic fact not explicitly supported by VERIFIED_RESULT or SOURCE_EVIDENCE.
2. Preserve course codes, course names, counts, credits, prerequisites, batches, semesters and other factual values exactly.
3. Never silently combine different academic batches or structures.
4. Never turn unknown/missing source values into real values.
5. Never invent recommendations, policies, eligibility rules, prerequisites, availability or requirements.
6. Use everyday language suitable for a person who is not a university student. If the word prerequisite is necessary, briefly explain that it means a course or requirement that must be completed first according to the record.
7. direct_answer must answer the question first.
8. simple_explanation must paraphrase the verified result, not add facts.
9. details may contain only useful facts explicitly present in VERIFIED_RESULT; otherwise use an empty list.
10. If status=follow_up, ask the clarification directly and explain why it is needed.
11. If status=insufficient, clearly say the supplied records are not enough.
12. If status=conflict, report the difference and do not choose a winner.
13. For counts, preserve the exact verified count and scope.
14. For lists, keep the list readable and do not add unsupported fields.
15. For a greeting, respond naturally and briefly.
16. Do not mention Gemini, LangChain, LangGraph, RAG, prompts, models, tools or internal processing.

STATUS:
{status}

USER QUESTION:
{query}

VERIFIED_RESULT:
{base.get('answer','')}

SOURCE_EVIDENCE:
{evidence_text or '(none)'}
'''
    obj,err=_call_gemini_json(prompt)
    if not isinstance(obj,dict):
        return _render_structured_response(_deterministic_plain_language(base)), err or "No structured response returned"
    for key in ["headline","direct_answer","simple_explanation","note"]: obj.setdefault(key,"")
    obj.setdefault("details",[])
    if not isinstance(obj.get("details"),list): obj["details"]=[]
    rendered=_render_structured_response(obj)
    if not rendered: rendered=_render_structured_response(_deterministic_plain_language(base))
    return rendered,err


def gemini_production_format(query, base, student_id=None):
    return gemini_structured_response(query, base, student_id)


def _answer_question_pipeline_legacy(query, student_id=None):
    t0=time.time()
    original_query=apply_conversation_context(query,rag["courses"])
    intent_info,intent_error=gemini_understand_query(original_query,student_id)

    # Deterministic planner is authoritative for source filtering. Gemini enriches it.
    det_plan=deterministic_query_plan(original_query,rag["minors"],rag["semesters"],rag["structures"])
    if intent_info.get("minor") and not det_plan.get("minor"): det_plan["minor"]=intent_info["minor"]
    if intent_info.get("years"): det_plan["years"]=intent_info["years"]
    if intent_info.get("semester"): det_plan["semester"]=intent_info["semester"]
    if intent_info.get("operation") and intent_info.get("operation")!="answer": det_plan["operation"]=intent_info["operation"]
    if intent_info.get("credits"): det_plan.setdefault("filters",{})["credits"]=intent_info["credits"]
    if isinstance(intent_info.get("filters"),dict): det_plan.setdefault("filters",{}).update({k:v for k,v in intent_info["filters"].items() if v not in (None,"",False)})

    retrieval_query=intent_info.get("retrieval_query") or original_query
    # Retrieval is run for provenance/context, while deterministic table verification decides facts.
    hits=retrieve(retrieval_query,rag)
    base=direct_answer(original_query,student_id,rag,plan=det_plan)

    found_codes=re.findall(r"\b[A-Z]{2,8}\d{3}\b",base.get("answer","").upper())
    if found_codes:
        for code in found_codes:
            if len(rag["courses"][rag["courses"].course_code.eq(code)])>0:
                st.session_state["last_course_code"]=code; break

    final,gemini_error=gemini_production_format(original_query,base,student_id)
    if not final: final=format_production_answer(base)
    combined_error=""
    if intent_error: combined_error += f"Intent: {intent_error}"
    if gemini_error: combined_error += (" | " if combined_error else "") + f"Response: {gemini_error}"
    return final,base,hits,time.time()-t0,combined_error


# ---------- LangGraph production orchestration ----------
class AIRAState(TypedDict, total=False):
    query: str
    student_id: str
    original_query: str
    intent_info: Dict[str, Any]
    intent_error: str
    det_plan: Dict[str, Any]
    generic_plan: Dict[str, Any]
    retrieval_query: str
    hits: List[Dict[str, Any]]
    base: Dict[str, Any]
    final: str
    gemini_error: str
    elapsed: float


def _graph_normalize(state: AIRAState):
    q = apply_conversation_context(state.get("query", ""), rag["courses"])
    return {"original_query": q}


def _graph_understand(state: AIRAState):
    info, err = gemini_understand_query(state["original_query"], state.get("student_id") or None)
    return {"intent_info": info, "intent_error": err or ""}


def _graph_plan(state: AIRAState):
    q = state["original_query"]
    info = state.get("intent_info") or {}
    # Open-ended semantic plan: Gemini may express arbitrary combinations of fields/filters.
    gplan,gerr=gemini_generic_plan(q,state.get("student_id") or None)
    plan = deterministic_query_plan(q, rag["minors"], rag["semesters"], rag["structures"])
    if info.get("minor") and not plan.get("minor"): plan["minor"] = info["minor"]
    if info.get("years"): plan["years"] = info["years"]
    if info.get("semester"): plan["semester"] = info["semester"]
    if info.get("operation") and info.get("operation") != "answer": plan["operation"] = info["operation"]
    if info.get("credits"): plan.setdefault("filters", {})["credits"] = info["credits"]
    if isinstance(info.get("filters"),dict): plan.setdefault("filters",{}).update({k:v for k,v in info["filters"].items() if v not in (None,"",False)})
    rq=(gplan or {}).get("retrieval_query") or info.get("retrieval_query") or q
    return {"det_plan": plan, "generic_plan": gplan or {}, "retrieval_query": rq, "intent_error": state.get("intent_error","") + ((" | Generic plan: "+gerr) if gerr else "")}

def _graph_retrieve(state: AIRAState):
    # LangChain chunks + hybrid FAISS/lexical retrieval provide context/provenance.
    hits = retrieve(state.get("retrieval_query") or state["original_query"], rag)
    return {"hits": hits}


def _graph_verify(state: AIRAState):
    # Policy documents are a first-class source type. Use them before the tabular executor
    # when the question is about university rules/SOP rather than course-table facts.
    ql=state["original_query"].lower()
    policy_terms=["attendance","registration","add/drop","add drop","add or drop","audit course","audit a course","summer term","digii","grievance","scholarship","parking","helmet","vehicle","speed limit","progression","transfer of credits","award of degree","award of a degree","bonafide","fee","late registration"]
    base = policy_document_fallback(state["original_query"], rag) if any(t in ql for t in policy_terms) else None
    if base is None:
        base = execute_generic_plan(state.get("generic_plan") or {}, rag, state["original_query"])
    if base is None:
        base = direct_answer(state["original_query"], state.get("student_id") or None, rag, plan=state.get("det_plan"))
    found_codes=re.findall(r"\b[A-Z]{2,8}\d{3}\b",str(base.get("answer","")).upper())
    if found_codes:
        for code in found_codes:
            if len(rag["courses"][rag["courses"].course_code.eq(code)])>0:
                st.session_state["last_course_code"] = code; break
    return {"base": base}

def _graph_generate(state: AIRAState):
    base = state.get("base", {})
    final, err = gemini_production_format(state["original_query"], base, state.get("student_id") or None)
    valid, validation_error = validate_llm_response(final, base, rag)
    if not valid:
        if validation_error:
            err = (err + " | " if err else "") + "Grounding guard: " + validation_error
        final = format_production_answer(base)
    if not final:
        final = format_production_answer(base)
    errors=[]
    if state.get("intent_error"): errors.append(f"Intent: {state['intent_error']}")
    if err: errors.append(f"Response: {err}")
    return {"final": final, "gemini_error": " | ".join(errors)}


@st.cache_resource(show_spinner=False)
def build_langgraph():
    """Build the production advisor graph once per Streamlit process."""
    if not LANGGRAPH_AVAILABLE:
        return None
    workflow = StateGraph(AIRAState)
    workflow.add_node("normalize", _graph_normalize)
    workflow.add_node("understand", _graph_understand)
    workflow.add_node("plan", _graph_plan)
    workflow.add_node("retrieve", _graph_retrieve)
    workflow.add_node("verify", _graph_verify)
    workflow.add_node("generate", _graph_generate)
    workflow.add_edge(START, "normalize")
    workflow.add_edge("normalize", "understand")
    workflow.add_edge("understand", "plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "verify")
    workflow.add_edge("verify", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


def answer_question(query, student_id=None):
    """Production entry point: Gemini + LangGraph + LangChain chunks + verified data."""
    t0=time.time()
    graph = build_langgraph()
    if graph is not None:
        try:
            state = graph.invoke({"query": query, "student_id": student_id or ""})
            final = state.get("final") or format_production_answer(state.get("base", {}))
            return final, state.get("base", {}), state.get("hits", []), time.time()-t0, state.get("gemini_error", "")
        except Exception as graph_error:
            # Never crash the student-facing app because of orchestration/runtime issues.
            # Fall back to the same grounded pipeline without exposing the traceback.
            result = _answer_question_pipeline_legacy(query, student_id)
            final, base, hits, elapsed, legacy_error = result
            msg = f"LangGraph fallback: {type(graph_error).__name__}: {graph_error}"
            if legacy_error: msg += " | " + legacy_error
            return final, base, hits, time.time()-t0, msg
    # Local dependency fallback; production requirements install LangGraph.
    return _answer_question_pipeline_legacy(query, student_id)

# ---------- Visual design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
html, body, [class*="css"]{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;}
.stApp{min-height:100vh;background:radial-gradient(circle at 8% 8%,rgba(124,92,255,.16),transparent 24%),radial-gradient(circle at 92% 14%,rgba(0,194,255,.13),transparent 25%),radial-gradient(circle at 50% 88%,rgba(47,211,154,.10),transparent 30%),linear-gradient(180deg,#fbfcff 0%,#f1f5ff 100%);}
.block-container{max-width:980px;padding:34px 28px 80px;}
.first-screen{min-height:72vh;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:30px;}
.orb-wrap{position:relative;width:132px;height:132px;display:flex;align-items:center;justify-content:center;}
.orb{width:92px;height:92px;border-radius:31px;position:relative;background:linear-gradient(145deg,#172b4d 0%,#315f9f 55%,#79a6ff 100%);box-shadow:0 24px 60px rgba(38,75,130,.22),inset 0 1px 0 rgba(255,255,255,.32);transform:rotate(-8deg);}
.orb:before,.orb:after{content:"";position:absolute;border-radius:50%;background:rgba(255,255,255,.92);}
.orb:before{width:13px;height:13px;left:28px;top:29px;box-shadow:26px 0 0 rgba(255,255,255,.92);}
.orb:after{width:30px;height:9px;left:31px;top:57px;border-radius:20px;background:rgba(255,255,255,.86);}
.ring{position:absolute;border:1px solid rgba(74,116,181,.16);border-radius:50%;}
.ring.r1{width:116px;height:116px}.ring.r2{width:145px;height:145px;opacity:.65}.ring.r3{width:176px;height:176px;opacity:.35}
div[data-testid="stChatInput"]{border-radius:24px!important;background:rgba(255,255,255,.94)!important;border:1px solid #dfe6ef!important;box-shadow:0 18px 48px rgba(30,53,84,.10)!important;padding:6px!important;}
div[data-testid="stChatInput"] textarea{font-size:1rem!important;color:#20324a!important;padding:13px 15px!important;}
div[data-testid="stChatInput"] textarea::placeholder{color:#a5afbd!important;}
.answer-page{padding-top:26px}.answer-shell{max-width:820px;margin:22px auto 0;padding:30px 32px;border:1px solid rgba(112,133,177,.18);border-radius:28px;background:linear-gradient(145deg,rgba(255,255,255,.98),rgba(248,250,255,.96));box-shadow:0 22px 70px rgba(38,63,110,.12);position:relative;overflow:hidden}.answer-shell:before{content:'';position:absolute;left:0;top:0;width:100%;height:5px;background:linear-gradient(90deg,#6c5ce7,#24b6ff,#32d39b);}.answer-kicker{display:flex;align-items:center;justify-content:space-between;margin-bottom:15px}.answer-kicker .who{font-size:.70rem;text-transform:uppercase;letter-spacing:.14em;color:#8794a7;font-weight:800}.answer-time{font-size:.70rem;color:#a0aaba;font-weight:600}.answer-text{font-size:1.05rem;line-height:1.78;color:#182b47}.answer-text h3{font-family:'Manrope',sans-serif;font-size:1.55rem;line-height:1.25;margin:0 0 20px;color:#14284a}.answer-text p{margin:.45rem 0 1rem}.answer-text li{margin:.38rem 0}.answer-text em{color:#75839a}.answer-text strong{color:#122b4d}
.status-pill{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:999px;font-size:.70rem;font-weight:700}.s-verified{background:#e8f6ee;color:#1a7f5a}.s-follow_up{background:#fff5df;color:#986b16}.s-insufficient{background:#eef1f5;color:#5b6472}.s-conflict{background:#fbeaea;color:#b23b3b}.s-greeting{background:#edf3fb;color:#365579}
.source-title{max-width:820px;margin:28px auto 10px;font-size:.72rem;text-transform:uppercase;letter-spacing:.16em;color:#667693;font-weight:800}.source-card{max-width:820px;margin:8px auto;padding:14px 16px;border:1px solid rgba(107,128,171,.16);border-radius:17px;background:rgba(255,255,255,.82);display:flex;gap:11px;align-items:flex-start;box-shadow:0 8px 24px rgba(50,72,110,.045)}.source-mark{color:#6c5ce7;font-size:1rem;font-weight:800}.source-label{font-size:.77rem;font-weight:700;color:#3d5b7f}.source-detail{font-size:.76rem;color:#758398;margin-top:3px;line-height:1.45}.follow{max-width:820px;margin:16px auto;padding:13px 16px;border-radius:16px;background:linear-gradient(135deg,#fff9e9,#fff4dc);border:1px solid #f2dfae;color:#805f19;font-size:.86rem}.ask-again{max-width:820px;margin:18px auto;text-align:center}.ask-again button{border-radius:14px!important;border:1px solid #dfe6ef!important;background:#fff!important}
section[data-testid="stSidebar"]{background:#fbfcff;border-right:1px solid #e7ebf1}[data-testid="stToolbar"]{visibility:hidden}
</style>
""",unsafe_allow_html=True)

profiles=["No student profile"]+[f"{r.student_id} · {r.minor} · Sem {r.semester}" for _,r in students.iterrows()]
with st.sidebar:
    st.markdown("### AIRA")
    selected=st.selectbox("Student profile",profiles)
    st.divider()
    st.caption("Academic facts come only from the supplied Excel workbooks. Gemini understands and writes the response; the verified records remain the source of truth.")
student_id=None if selected=="No student profile" else selected.split(" · ")[0]

STATUS_META={"verified":("✓","Verified"),"follow_up":("◌","Needs more info"),"insufficient":("○","Not in records"),"conflict":("▲","Conflicting records"),"greeting":("●","Hello"),"conversation":("✦","AIRA")}
pending=st.session_state.pop("pending_question",None)
q=st.chat_input(" ")
query=q or pending

if not query:
    st.markdown('<div class="first-screen"><div class="orb-wrap"><div class="ring r1"></div><div class="ring r2"></div><div class="ring r3"></div><div class="orb"></div></div></div>',unsafe_allow_html=True)
else:
    with st.spinner(" "):
        final,base,hits,elapsed,gemini_error=answer_question(query,student_id)
    mark,label=STATUS_META.get(base.get("status"),("●","Answer"))
    st.markdown('<div class="answer-page">',unsafe_allow_html=True)
    st.markdown(f'<div class="answer-shell"><div class="answer-kicker"><span class="who">AIRA</span><span><span class="status-pill s-{base.get("status","verified")}">{mark} {label}</span> <span class="answer-time">{elapsed:.1f}s</span></span></div><div class="answer-text">',unsafe_allow_html=True)
    st.markdown(final)
    st.markdown('</div></div>',unsafe_allow_html=True)
    source_items=base.get("evidence",[])
    if source_items:
        st.markdown('<div class="source-title">Sources</div>',unsafe_allow_html=True); render_sources(source_items)
    if base.get("status")=="follow_up": st.markdown('<div class="follow">◌ Please clarify the missing detail so AIRA can answer without guessing.</div>',unsafe_allow_html=True)
    st.markdown('<div class="ask-again">',unsafe_allow_html=True)
    if st.button("＋  Ask another question",use_container_width=False): st.rerun()
    st.markdown('</div></div>',unsafe_allow_html=True)


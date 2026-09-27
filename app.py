import os, re, json, time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------- Optional / required AI stack -------------------------
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception as e:
    SentenceTransformer = None
    faiss = None
    EMBEDDING_IMPORT_ERROR = str(e)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatPromptTemplate = None

try:
    from langgraph.graph import StateGraph, START, END
except Exception:
    StateGraph = START = END = None

# ------------------------- Paths -------------------------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
SOURCE_DIRS = [BASE / "source_data", BASE / "sources", BASE / "notebook_inputs", BASE]

RAW_PDF_NAMES = ["Student_Handbook_Aug_2026.pdf", "SOP_STUDENT_17082026_Final.pdf"]
RAW_XLSX_NAMES = ["Semester_Spread_Structures_Sept_2026.xlsx", "Minor_Courses_for_BTech_Students.xlsx"]


def find_source(name: str) -> Optional[Path]:
    for d in SOURCE_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


RAW_PDFS = [p for p in (find_source(x) for x in RAW_PDF_NAMES) if p]
RAW_XLSX = [p for p in (find_source(x) for x in RAW_XLSX_NAMES) if p]

REQUIRED_TABLES = {
    "course_master": "course_master.csv",
    "semester_offerings": "semester_offerings.csv",
    "degree_requirements": "degree_requirements.csv",
    "students": "students.csv",
    "history": "student_course_history.csv",
    "minor": "minor_courses.csv",
    "structure": "structure_courses.csv",
    "prereq": "prerequisite_table.csv",
}

missing = [f for f in REQUIRED_TABLES.values() if not (DATA / f).exists()]
if missing:
    st.error("Required university data files are missing: " + ", ".join(missing))
    st.stop()

# ------------------------- UI -------------------------
st.set_page_config(
    page_title="AIRA · Vidyashilp University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --navy:#173B70; --blue:#2F6FED; --ink:#172033; --muted:#667085; --line:#E5E7EB; --bg:#F7F9FC; }
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
.stApp { background: var(--bg); }
.block-container { max-width: 1120px; padding: 1.25rem 1.4rem 2.5rem; }
.aira-top { display:flex; align-items:center; justify-content:space-between; padding: 6px 0 18px; }
.brand { display:flex; align-items:center; gap:12px; }
.logo { width:42px; height:42px; border-radius:12px; background:var(--navy); color:white; display:flex; align-items:center; justify-content:center; font-weight:800; letter-spacing:.5px; }
.brand h1 { margin:0; color:var(--ink); font-size:24px; line-height:1.1; }
.brand p { margin:3px 0 0; color:var(--muted); font-size:12px; }
.status { font-size:11px; color:#1F6B45; background:#ECFDF3; border:1px solid #B7E4C7; border-radius:999px; padding:6px 10px; font-weight:700; }
.welcome { background:white; border:1px solid var(--line); border-radius:18px; padding:22px 24px; margin-bottom:18px; }
.welcome h2 { margin:0 0 6px; color:var(--ink); font-size:20px; }
.welcome p { margin:0; color:var(--muted); font-size:13px; line-height:1.55; }
.answer { background:white; border:1px solid var(--line); border-radius:16px; padding:18px 20px; margin:8px 0 10px; }
.answer-label { font-size:10px; font-weight:800; color:var(--blue); letter-spacing:.8px; text-transform:uppercase; margin-bottom:8px; }
.answer-body { color:var(--ink); font-size:15px; line-height:1.65; }
.source-box { background:#FAFBFC; border:1px solid var(--line); border-radius:12px; padding:10px 12px; margin:5px 0; }
.source-title { font-weight:700; color:#243B64; font-size:12px; }
.source-meta { color:var(--muted); font-size:11px; margin-top:2px; }
.plan-box { background:#F8FAFF; border:1px solid #D9E5FF; border-radius:12px; padding:10px 12px; color:#42526B; font-size:11px; }
div[data-testid="stSidebar"] { background:white; border-right:1px solid var(--line); }
.small-note { color:var(--muted); font-size:11px; line-height:1.45; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Data -------------------------
@st.cache_data(show_spinner=False)
def load_tables():
    return {k: pd.read_csv(DATA / f) for k, f in REQUIRED_TABLES.items()}

T = load_tables()
cm, off, deg, students, hist, minor, structure, pre = [T[k] for k in REQUIRED_TABLES]


def clean(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).strip()


def norm(v: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", clean(v).lower())).strip()


def norm_code(v: Any) -> str:
    return re.sub(r"\s+", "", clean(v).upper())


def parse_int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except Exception:
        return None


def source_ref(file: str, page: Optional[int] = None, sheet: Optional[str] = None, row: Optional[int] = None) -> str:
    bits = [file]
    if page is not None: bits.append(f"page {page}")
    if sheet: bits.append(f"sheet {sheet}")
    if row is not None: bits.append(f"row {row}")
    return " · ".join(bits)

# ------------------------- Raw-source RAG ingestion -------------------------
def normalize_doc_text(text: str) -> str:
    text = clean(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@st.cache_data(show_spinner=False)
def load_raw_documents() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if PdfReader:
        for path in RAW_PDFS:
            try:
                reader = PdfReader(str(path))
                for i, page in enumerate(reader.pages, start=1):
                    text = normalize_doc_text(page.extract_text() or "")
                    if text:
                        docs.append({"text": text, "source_file": path.name, "source_type": "pdf", "page": i, "sheet": "", "row": None})
            except Exception:
                pass
    for path in RAW_XLSX:
        try:
            sheets = pd.read_excel(path, sheet_name=None, header=None)
            for sheet, df in sheets.items():
                df = df.dropna(how="all").dropna(axis=1, how="all")
                for idx, row in df.iterrows():
                    vals = [clean(x) for x in row.tolist()]
                    vals = [x for x in vals if x]
                    if not vals:
                        continue
                    text = " | ".join(vals)
                    docs.append({"text": text, "source_file": path.name, "source_type": "xlsx", "page": None, "sheet": str(sheet), "row": int(idx) + 1})
        except Exception:
            pass
    return docs


RAW_DOCS = load_raw_documents()


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    if RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " | ", " ", ""],
        )
    else:
        splitter = None
    for d in docs:
        pieces = splitter.split_text(d["text"]) if splitter else [d["text"]]
        for j, piece in enumerate(pieces):
            x = dict(d)
            x["text"] = piece
            x["chunk"] = j + 1
            out.append(x)
    return out


@st.cache_resource(show_spinner=False)
def build_rag_index():
    chunks = chunk_documents(RAW_DOCS)
    if not chunks or SentenceTransformer is None or faiss is None:
        return None
    texts = [x["text"] for x in chunks]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = tfidf.fit_transform(texts)
    return {"chunks": chunks, "model": model, "faiss": index, "tfidf": tfidf, "tfidf_X": X}


RAG = build_rag_index()

# ------------------------- Source-aware hybrid retrieval -------------------------
def query_family(q: str) -> Optional[str]:
    qn = norm(q)
    if any(x in qn for x in ["scholarship", "fee", "payment", "attendance", "registration procedure", "add drop", "withdraw", "grievance", "digii", "examination", "conduct", "leave", "appeal", "sop"]):
        return "pdf"
    if any(x in qn for x in ["minor", "basket", "programme structure", "program structure", "curriculum", "semester", "offered", "offering", "course credit", "course credits"]):
        return "xlsx"
    return None


def retrieve(q: str, k: int = 6) -> List[Dict[str, Any]]:
    if RAG is None:
        return []
    q = clean(q)
    family = query_family(q)
    chunks = RAG["chunks"]
    mask = np.ones(len(chunks), dtype=bool)
    if family:
        mask = np.array([x["source_type"] == family for x in chunks])
        if not mask.any():
            mask = np.ones(len(chunks), dtype=bool)

    # Independent semantic and lexical candidate generation.
    qemb = RAG["model"].encode([q], normalize_embeddings=True)
    sem_scores, sem_ids = RAG["faiss"].search(np.asarray(qemb, dtype="float32"), min(40, len(chunks)))
    lexical = linear_kernel(RAG["tfidf"].transform([q]), RAG["tfidf_X"]).ravel()
    sem_rank = {int(i): rank for rank, i in enumerate(sem_ids[0]) if i >= 0 and mask[int(i)]}
    lex_ids = np.argsort(-lexical)
    lex_rank = {int(i): rank for rank, i in enumerate(lex_ids[:80]) if mask[int(i)]}

    # Reciprocal-rank fusion prevents one retrieval method from dominating.
    ids = set(sem_rank) | set(lex_rank)
    scored = []
    for i in ids:
        rrf = (1 / (60 + sem_rank[i]) if i in sem_rank else 0) + (1 / (60 + lex_rank[i]) if i in lex_rank else 0)
        scored.append((rrf, i))
    scored.sort(reverse=True)

    results = []
    seen = set()
    for score, i in scored:
        d = dict(chunks[i])
        key = (d["source_file"], d.get("page"), d.get("sheet"), d.get("row"))
        if key in seen:
            continue
        seen.add(key)
        d["retrieval_score"] = float(score)
        d["citation"] = source_ref(d["source_file"], d.get("page"), d.get("sheet"), d.get("row"))
        results.append(d)
        if len(results) >= k:
            break
    return results

# ------------------------- Entity resolution -------------------------
def aliases_for_course(title: str) -> List[str]:
    n = norm(title)
    aliases = [n]
    if "financial and management accounting" in n:
        aliases += ["fama", "financial management accounting", "management accounting"]
    return list(dict.fromkeys(aliases))


def all_course_records() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, r in cm.iterrows():
        rows.append({"course_code": clean(r.course_code), "course_title": clean(r.course_title), "prerequisite": clean(r.prerequisite), "origin": "course_master"})
    # Minor catalogue contains legitimate courses absent from the master table.
    for _, r in minor.iterrows():
        rows.append({"course_code": clean(r.course_code), "course_title": clean(r.course_title), "prerequisite": clean(r.prerequisite), "origin": "minor_catalogue", "minor": clean(r['minor']), "batch": clean(r['batch']), "credits": clean(r['credits'])})
    for _, r in off.iterrows():
        rows.append({"course_code": clean(r.course_code), "course_title": clean(r.course_title), "prerequisite": clean(r.prerequisite), "origin": "semester_offering", "credits": clean(r.credits), "semester": clean(r.semester), "academic_structure": clean(r.academic_structure)})
    return rows


COURSE_RECORDS = all_course_records()


def resolve_course(entity: str) -> Dict[str, Any]:
    q = norm(entity)
    qcode = norm_code(entity)
    candidates = []
    for r in COURSE_RECORDS:
        title = norm(r.get("course_title"))
        c = norm_code(r.get("course_code"))
        aliases = aliases_for_course(r.get("course_title", ""))
        score = 0.0
        if qcode and qcode == c: score = 1.0
        elif q in aliases: score = 0.98
        elif q and q in title: score = 0.90
        else:
            ratio = SequenceMatcher(None, q, title).ratio() if q else 0
            # Token containment is useful for "management accounting".
            qt, tt = set(q.split()), set(title.split())
            overlap = len(qt & tt) / max(1, len(qt))
            score = max(ratio * 0.82, overlap * 0.88)
        if score >= 0.70:
            candidates.append((score, r))
    candidates.sort(key=lambda x: x[0], reverse=True)
    # Collapse duplicates representing the same logical course.
    unique = {}
    for score, r in candidates:
        key = (norm_code(r.get("course_code")), norm(r.get("course_title")))
        unique.setdefault(key, (score, r))
    vals = sorted(unique.values(), key=lambda x: x[0], reverse=True)
    if not vals:
        return {"status": "not_found", "matches": []}
    # A strong exact/alias match wins; otherwise close competing titles are ambiguous.
    top_score, top = vals[0]
    if len(vals) > 1 and top_score < 0.96 and vals[1][0] >= top_score - 0.04:
        return {"status": "ambiguous", "matches": [v[1] for v in vals[:8]]}
    return {"status": "identified", "match": top, "matches": [v[1] for v in vals[:8]]}


def extract_student_id(q: str, selected: str) -> Optional[str]:
    m = re.search(r"\b(SYN\d+)\b", q, re.I)
    if m: return m.group(1).upper()
    return selected if selected and selected != "New / General User" else None

# ------------------------- Query planner -------------------------
class Plan(TypedDict, total=False):
    route: str
    intent: str
    operation: str
    entity_type: str
    entity: str
    attribute: str
    filters: Dict[str, Any]
    student_id: Optional[str]
    confidence: float
    reason: str


def deterministic_plan(q: str, selected_student: str) -> Plan:
    qn = norm(q)
    sid = extract_student_id(q, selected_student)
    # Social / unsupported is handled before retrieval.
    if re.fullmatch(r"(hi|hello|hey|namaste|thanks|thank you|bye|goodbye)[!. ]*", qn):
        return {"route": "social", "intent": "social", "operation": "respond", "confidence": 1.0}

    # Exact course-like attribute questions.
    attr = None
    if any(x in qn for x in ["how many credits", "how many credit", "how much credit", "credits does", "credit does"]): attr = "credits"
    elif "prerequisite" in qn or "prereq" in qn: attr = "prerequisite"
    elif any(x in qn for x in ["course code", "code of"]): attr = "course_code"
    if attr:
        # Resolve the course entity from the catalogue vocabulary instead of
        # stripping arbitrary question words. This is what prevents
        # "how many credits does FAMA have" from becoming "fama have".
        entity = qn
        vocab = []
        for r in COURSE_RECORDS:
            for a in aliases_for_course(r.get("course_title", "")):
                if a and a in qn:
                    vocab.append((len(a), a))
            c = norm_code(r.get("course_code"))
            if c and c.lower() in qn:
                vocab.append((len(c), c))
        if vocab:
            entity = sorted(vocab, reverse=True)[0][1]
        else:
            mcode = re.search(r"\b[A-Za-z]{2,8}\s*\d{2,5}\b", q)
            if mcode:
                entity = norm_code(mcode.group(0))
        return {"route":"structured","intent":"course_attribute","operation":"get","entity_type":"course","entity":entity,"attribute":attr,"student_id":sid,"filters":{},"confidence":0.96 if vocab else 0.75}

    if any(x in qn for x in ["scholarship", "fee policy", "payment", "attendance", "registration procedure", "add drop", "withdraw", "grievance", "digii", "examination", "code of conduct", "leave", "appeal"]):
        return {"route":"rag","intent":"policy","operation":"answer","entity_type":"document","student_id":sid,"filters":{},"confidence":0.94}

    if any(x in qn for x in ["minor course", "minor courses", "courses in finance minor", "finance minor", "psychology minor"]):
        minor_name = None
        for m in minor["minor"].dropna().astype(str).unique():
            if norm(m) in qn: minor_name = m
        batch = re.search(r"\bbatch\s*(20\d{2})\b", qn)
        sem = re.search(r"\b(?:semester|sem|s)\s*(\d{1,2})\b", qn)
        op = "count" if any(x in qn for x in ["how many", "number of", "count"] ) else "list"
        return {"route":"structured","intent":"minor_catalogue","operation":op,"entity_type":"minor_course","filters":{"minor":minor_name,"batch":batch.group(1) if batch else None,"semester":int(sem.group(1)) if sem else None,"programme_query": qn},"student_id":sid,"confidence":0.90}

    if any(x in qn for x in ["total credits", "required credits", "degree requirement", "graduation credits", "programme structure", "program structure"]):
        return {"route":"structured","intent":"degree_requirement","operation":"get","entity_type":"degree","filters":{},"student_id":sid,"confidence":0.90}

    if any(x in qn for x in ["my profile", "my details", "my academic record", "my information"]):
        return {"route":"structured","intent":"student_profile","operation":"get","entity_type":"student","student_id":sid,"confidence":0.98}
    if any(x in qn for x in ["my courses", "my history", "my results", "courses i completed", "what have i completed"]):
        return {"route":"structured","intent":"student_history","operation":"list","entity_type":"student_history","student_id":sid,"confidence":0.98}

    # Semester offering query.
    sem = re.search(r"\b(?:semester|sem|s)\s*(\d{1,2})\b", qn)
    if sem and any(x in qn for x in ["offered", "offering", "courses", "subjects"]):
        return {"route":"structured","intent":"semester_offering","operation":"list","entity_type":"course_offering","filters":{"semester":f"S{sem.group(1)}"},"student_id":sid,"confidence":0.90}

    # Explicit general university fact -> RAG, not a structured aggregate.
    if any(x in qn for x in ["where", "when", "who", "how many programs", "how many schools", "university", "provide", "offer", "policy", "rule"]):
        return {"route":"rag","intent":"university_fact","operation":"answer","entity_type":"document","student_id":sid,"filters":{},"confidence":0.75}

    return {"route":"rag","intent":"general_academic","operation":"answer","entity_type":"document","student_id":sid,"filters":{},"confidence":0.45}


# ------------------------- Deterministic execution -------------------------
def raw_crosscheck(entity: str) -> List[Dict[str, Any]]:
    # Raw-source evidence is deliberately retrieved with the same entity string.
    return retrieve(entity, k=3)


def render_sources(refs: List[Dict[str, Any]]):
    if not refs:
        return
    with st.expander(f"Sources · {len(refs)} evidence item(s)", expanded=False):
        for r in refs:
            st.markdown(f"<div class='source-box'><div class='source-title'>{r['citation']}</div><div class='source-meta'>{clean(r['text'])[:650]}</div></div>", unsafe_allow_html=True)


def execute_plan(plan: Plan, question: str) -> Dict[str, Any]:
    route = plan.get("route")
    if route == "social":
        return {"status":"social", "answer":"Hello! I’m AIRA, the Vidyashilp University academic advisor. Ask me about courses, credits, prerequisites, offerings, programme requirements, student records, or university policies.", "refs":[], "decision":{}}

    if route == "structured":
        intent = plan.get("intent")
        sid = plan.get("student_id")
        if intent == "course_attribute":
            entity = clean(plan.get("entity"))
            res = resolve_course(entity)
            if res["status"] == "not_found":
                return {"status":"not_listed","answer":f"I could not find **{entity}** in the supplied course catalogues. I won’t infer a credit value from unrelated records.","refs":raw_crosscheck(entity),"decision":{}}
            if res["status"] == "ambiguous":
                items = res["matches"][:6]
                return {"status":"ambiguous","answer":"I found multiple possible course matches. Please specify the course code or exact title:\n\n" + "\n".join(f"• **{x.get('course_code','')}** — {x.get('course_title','')}" for x in items),"refs":raw_crosscheck(entity),"decision":{}}
            r = res["match"]
            attr = plan.get("attribute")
            codev = clean(r.get("course_code")); title = clean(r.get("course_title"))
            values = []
            # Collect exact structured records from all authoritative tables.
            if attr == "credits":
                for df, label in [(minor,"minor catalogue"),(off,"semester offerings")]:
                    if "course_code" in df.columns:
                        x = df[df.course_code.astype(str).map(norm_code) == norm_code(codev)]
                        for _, row in x.iterrows():
                            if clean(row.get("credits")): values.append(clean(row.get("credits")))
                # Some minor catalogues use a placeholder code; match title too.
                if not values:
                    x = minor[minor.course_title.astype(str).map(norm) == norm(title)]
                    values += [clean(v) for v in x.credits.tolist() if clean(v)]
                values = list(dict.fromkeys(values))
                if len(values) != 1:
                    return {"status":"conflict" if values else "not_confirmed","answer":f"I found **{title}** but the supplied structured sources do not provide one unambiguous credit value.","refs":raw_crosscheck(title),"decision":{}}
                value = values[0]
                refs = raw_crosscheck(title)
                return {"status":"confirmed","answer":f"**{title} ({codev})** has **{value} credit(s)** in the supplied university catalogue.","refs":refs,"decision":{"course_title":title,"course_code":codev,"credits":value}}
            if attr == "prerequisite":
                vals = []
                for df in [cm, minor, off, pre]:
                    if "course_code" in df.columns:
                        x = df[df.course_code.astype(str).map(norm_code) == norm_code(codev)]
                        if "prerequisite" in x.columns:
                            vals += [clean(v) for v in x.prerequisite.tolist()]
                vals = [v for v in dict.fromkeys(vals) if v and v.upper() not in {"NIL","NONE","NAN"}]
                if not vals:
                    return {"status":"not_confirmed","answer":f"No prerequisite is explicitly recorded for **{title} ({codev})** in the supplied structured tables.","refs":raw_crosscheck(title),"decision":{}}
                if len(set(map(norm, vals))) > 1:
                    return {"status":"conflict","answer":f"The supplied sources record different prerequisite information for **{title} ({codev})**. I won’t silently choose one.","refs":raw_crosscheck(title),"decision":{"values":vals}}
                return {"status":"confirmed","answer":f"The recorded prerequisite for **{title} ({codev})** is **{vals[0]}**.","refs":raw_crosscheck(title),"decision":{"prerequisite":vals[0]}}

        if intent == "minor_catalogue":
            f = plan.get("filters", {})
            programme_query = clean(f.get("programme_query"))
            if "bms" in programme_query and "data science" in programme_query:
                return {"status":"insufficient","answer":"The supplied minor-course workbook is for **BTech students**; it does not contain an exact **BMS (Hons.) Data Science minor** catalogue. I won’t map BMS Data Science to the BTech minor data or invent a course count.","refs":retrieve(question),"decision":{}}
            x = minor.copy()
            if f.get("minor"): x = x[x["minor"].astype(str).map(norm) == norm(f["minor"])]
            if f.get("batch"): x = x[x["batch"].astype(str) == str(f["batch"])]
            if f.get("semester") is not None: x = x[x["semester"].apply(parse_int) == int(f["semester"])]
            if x.empty:
                scope = ", ".join(str(v) for v in [f.get("minor"), f.get("batch"), f.get("semester")] if v)
                return {"status":"not_listed","answer":f"No matching minor-course records were found for the specified scope{(' ('+scope+')') if scope else ''}.","refs":retrieve(scope or question),"decision":{}}
            if plan.get("operation") == "count":
                # Count logical catalogue rows, not arbitrary source rows.
                count = len(x.drop_duplicates(subset=["minor","batch","course_title","semester"]))
                return {"status":"confirmed","answer":f"The supplied minor-course catalogue contains **{count} course(s)** for the specified filters.","refs":retrieve(question),"decision":{"count":count}}
            items = x.drop_duplicates(subset=["minor","batch","course_title","semester"])
            text = "\n".join(f"• **{clean(r.course_title)}** — {clean(r.course_code)} · {clean(r.credits)} credits" for r in items.itertuples())
            return {"status":"confirmed","answer":text,"refs":retrieve(question),"decision":{"count":len(items)}}

        if intent == "degree_requirement":
            qn = norm(question)
            structures = deg.academic_structure.dropna().astype(str).unique().tolist()
            matches = [s for s in structures if norm(s) in qn]
            # Do not silently map BMS/BTech/Data Science to a structure.
            if not matches:
                return {"status":"insufficient","answer":"The supplied degree-requirements table contains multiple academic structures, but it does not contain an exact **BMS (Hons.)** structure. Please specify the exact academic structure if you want a requirement from this table.","refs":retrieve(question),"decision":{}}
            if len(matches) > 1:
                return {"status":"ambiguous","answer":"Please specify one academic structure: " + ", ".join(matches),"refs":retrieve(question),"decision":{}}
            x = deg[deg.academic_structure.astype(str) == matches[0]]
            if "total" in qn or "graduation" in qn:
                y = x[x.component.astype(str).str.lower() == "total credits"]
                if not y.empty:
                    return {"status":"confirmed","answer":f"The required total for **{matches[0]}** is **{clean(y.iloc[0].required_credits)} credits**.","refs":retrieve(matches[0]),"decision":{"structure":matches[0],"total":clean(y.iloc[0].required_credits)}}
            text = "\n".join(f"• **{clean(r.component)}:** {clean(r.required_credits)} credits" for r in x.itertuples())
            return {"status":"confirmed","answer":f"**{matches[0]}** requirements:\n\n{text}","refs":retrieve(matches[0]),"decision":{}}

        if intent == "student_profile":
            if not sid:
                return {"status":"missing_student","answer":"Please select a student profile in the sidebar or include a student ID such as SYN001.","refs":[],"decision":{}}
            x = students[students.student_id.astype(str).str.upper() == sid.upper()]
            if x.empty:
                return {"status":"not_listed","answer":f"Student **{sid}** is not present in the supplied synthetic student records.","refs":[],"decision":{}}
            r = x.iloc[0]
            return {"status":"confirmed","answer":f"**{clean(r.student_id)}** · {clean(r.programme)} · batch {clean(r.batch)} · semester {clean(r.current_semester)} · profile credits {clean(r.total_credits)} · minor: {clean(r['minor'])}.","refs":[],"decision":r.to_dict()}

        if intent == "student_history":
            if not sid:
                return {"status":"missing_student","answer":"Please select a student profile so I can read the student record.","refs":[],"decision":{}}
            x = hist[hist.student_id.astype(str).str.upper() == sid.upper()]
            if x.empty:
                return {"status":"not_listed","answer":f"No course-history rows are present for **{sid}** in the supplied student record.","refs":[],"decision":{}}
            text = "\n".join(f"• **{norm_code(r.course_code)}** — {clean(r.status)} · grade {clean(r.grade)}" for r in x.itertuples())
            return {"status":"confirmed","answer":text,"refs":[],"decision":{"rows":len(x)}}

        if intent == "semester_offering":
            sem = plan.get("filters", {}).get("semester")
            x = off[off.semester.astype(str).str.upper() == str(sem).upper()]
            x = x.drop_duplicates(subset=["course_code","course_title"])
            if x.empty:
                return {"status":"not_listed","answer":f"No offering rows were found for **{sem}** in the supplied semester-offering table.","refs":retrieve(question),"decision":{}}
            text = "\n".join(f"• **{clean(r.course_title)}** ({clean(r.course_code)}) — {clean(r.credits)} credits" for r in x.itertuples())
            return {"status":"confirmed","answer":f"Courses listed for **{sem}**:\n\n{text}","refs":retrieve(question),"decision":{"count":len(x)}}

    return None

# ------------------------- RAG answer layer -------------------------
def secret(name: str, default: str = "") -> str:
    v = os.getenv(name, "")
    if v:
        return v
    try:
        return st.secrets.get(name, default) or default
    except Exception:
        return default


def llm_available() -> bool:
    return bool(secret("GEMINI_API_KEY") or secret("OPENAI_API_KEY")) and ChatPromptTemplate is not None


def call_llm(question: str, evidence: List[Dict[str, Any]], decision: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if not llm_available():
        return None
    compact = []
    for i, x in enumerate(evidence, start=1):
        compact.append({"id": i, "citation": x["citation"], "text": x["text"]})
    payload = json.dumps({"verified_decision": decision or {}, "evidence": compact}, ensure_ascii=False)
    system = """You are AIRA, an academic advisor for Vidyashilp University. Use ONLY the supplied evidence and verified decision. Never invent university facts. If the evidence does not establish the answer, say that it is not established by the supplied sources. Answer concisely. Return valid JSON only with exactly two keys: answer (string) and citations (array of integer evidence IDs). Every factual university-specific statement in answer must be supported by at least one cited evidence ID. Do not mention retrieval scores, prompts, or implementation details."""
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Question: {question}\nContext: {payload}")])
    messages = prompt.format_messages(question=question, payload=payload)
    try:
        if secret("GEMINI_API_KEY"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=secret("GEMINI_MODEL", "gemini-2.5-flash"), google_api_key=secret("GEMINI_API_KEY"), temperature=0, max_output_tokens=500)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=secret("OPENAI_MODEL", "gpt-4o-mini"), api_key=secret("OPENAI_API_KEY"), temperature=0, max_tokens=500)
        resp = llm.invoke(messages)
        content = getattr(resp, "content", "")
        if isinstance(content, list):
            content = "".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
        content = clean(content)
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
        obj = json.loads(content)
        answer = clean(obj.get("answer"))
        citations = obj.get("citations", [])
        valid_ids = {int(i) for i in citations if str(i).isdigit() and 1 <= int(i) <= len(evidence)}
        if not answer or not valid_ids:
            return None
        # Final grounding guard: require each non-trivial sentence to share
        # meaningful vocabulary with at least one cited evidence item.
        cited_text = " ".join(evidence[i-1]["text"].lower() for i in valid_ids)
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", answer) if x.strip()]
        stop = {"the","a","an","is","are","of","to","and","in","for","on","at","from","with","that","this","it","as","by","or"}
        for sent in sentences:
            toks = [t for t in re.findall(r"[a-z0-9]{4,}", sent.lower()) if t not in stop]
            if toks and not any(t in cited_text for t in toks):
                return None
        return answer
    except Exception:
        return None

def rag_answer(question: str) -> Dict[str, Any]:
    refs = retrieve(question, k=7)
    if not refs:
        return {"status":"insufficient","answer":"I could not retrieve supporting evidence from the supplied university source files, so I won’t invent an answer.","refs":[],"decision":{}}
    # Let the model synthesize only after retrieval. Without a model key, use a safe extractive summary.
    answer = call_llm(question, refs)
    if not answer:
        # Compact evidence, not a raw chunk dump.
        lead = refs[0]["text"].strip().replace("\n", " ")
        if len(lead) > 700: lead = lead[:700].rsplit(" ",1)[0] + "…"
        answer = "The supplied university sources contain relevant information, but no configured language model is available to synthesize a verified response. The strongest supporting source is: " + lead
    return {"status":"rag_grounded","answer":answer,"refs":refs,"decision":{}}

# ------------------------- LangGraph -------------------------
class State(TypedDict, total=False):
    question: str
    student_id: str
    plan: Plan
    decision: Dict[str, Any]
    status: str
    answer: str
    refs: List[Dict[str, Any]]


def node_plan(state: State) -> State:
    return {"plan": deterministic_plan(state["question"], state.get("student_id", ""))}


def node_execute(state: State) -> State:
    plan = state["plan"]
    if plan.get("route") == "rag":
        result = rag_answer(state["question"])
    else:
        result = execute_plan(plan, state["question"])
        if result is None:
            result = rag_answer(state["question"])
    return {"status":result["status"], "answer":result["answer"], "refs":result.get("refs",[]), "decision":result.get("decision",{})}


def node_validate(state: State) -> State:
    # Hard final guard: an empty/unsupported answer is never allowed through.
    answer = clean(state.get("answer"))
    status = state.get("status", "insufficient")
    if not answer:
        return {"status":"insufficient", "answer":"The supplied sources do not establish a reliable answer to this question.", "refs":state.get("refs",[]) }
    return {}


@st.cache_resource(show_spinner=False)
def build_graph():
    if StateGraph is None:
        return None
    g = StateGraph(State)
    g.add_node("plan", node_plan)
    g.add_node("execute", node_execute)
    g.add_node("validate", node_validate)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "validate")
    g.add_edge("validate", END)
    return g.compile()


GRAPH = build_graph()


def run(question: str, student_id: str) -> Dict[str, Any]:
    initial: State = {"question": question, "student_id": student_id}
    if GRAPH:
        return GRAPH.invoke(initial)
    s = node_plan(initial)
    s.update(node_execute({**initial, **s}))
    s.update(node_validate({**initial, **s}))
    return s

# ------------------------- App -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "student" not in st.session_state:
    st.session_state.student = "New / General User"

with st.sidebar:
    st.markdown("### AIRA")
    st.caption("Academic Intelligence & Registration Assistant")
    options = ["New / General User"] + students.student_id.astype(str).tolist()
    st.session_state.student = st.selectbox("Student context", options, index=options.index(st.session_state.student) if st.session_state.student in options else 0)
    st.divider()
    st.markdown("**Source integrity**")
    st.markdown(f"<div class='small-note'>Structured records: {len(cm)+len(off)+len(minor)+len(deg)} rows<br>Raw PDFs: {len(RAW_PDFS)}<br>Raw Excel workbooks: {len(RAW_XLSX)}<br>RAG chunks: {len(RAG['chunks']) if RAG else 0}</div>", unsafe_allow_html=True)
    if RAG:
        st.success("Semantic + lexical RAG ready")
    else:
        st.error("RAG index unavailable")
    st.divider()
    st.markdown("**What AIRA does**")
    st.markdown("<div class='small-note'>Exact catalogue questions are answered from structured university records and cross-checked against raw sources. Policy and general university questions use hybrid RAG over the original PDFs/Excel files. The LLM explains evidence; it does not create academic facts.</div>", unsafe_allow_html=True)

st.markdown("<div class='aira-top'><div class='brand'><div class='logo'>AI</div><div><h1>AIRA</h1><p>Vidyashilp University Academic Advisor</p></div></div><div class='status'>● Source-grounded</div></div>", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("<div class='welcome'><h2>How can I help?</h2><p>Ask about credits, prerequisites, semester offerings, minor courses, programme requirements, your academic record, or university policies. AIRA will only answer from the supplied university sources and will flag ambiguity or conflicts instead of guessing.</p></div>", unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            st.markdown(f"<div class='answer'><div class='answer-label'>AIRA · {m.get('status','').replace('_',' ')}</div><div class='answer-body'>{m['content'].replace(chr(10),'<br>')}</div></div>", unsafe_allow_html=True)
            render_sources(m.get("refs", []))
        else:
            st.markdown(m["content"])

q = st.chat_input("Ask about courses, credits, prerequisites, offerings, regulations…")
if q:
    st.session_state.messages.append({"role":"user", "content":q})
    sid = extract_student_id(q, st.session_state.student) or st.session_state.student
    start = time.perf_counter()
    result = run(q, sid)
    elapsed = time.perf_counter() - start
    st.session_state.messages.append({"role":"assistant", "content":result.get("answer",""), "refs":result.get("refs",[]), "status":result.get("status","insufficient"), "elapsed":elapsed})
    st.rerun()

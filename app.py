
import os, re, time, math, hashlib
from pathlib import Path
import pandas as pd
import streamlit as st
import openpyxl

# ============================================================
# AIRA — AI Academic Advisor | Assignment #1
# Authoritative sources: the two supplied Excel workbooks.
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

PASS_GRADES = {"A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-"}
FAIL_GRADES = {"F","FAIL","FAILED"}

# ---------- data cleaning & organization ----------
def norm(x):
    if x is None or (isinstance(x,float) and pd.isna(x)): return ""
    return re.sub(r"\s+"," ",str(x).replace("\xa0"," ").replace("’","'").strip())

def norm_code(x):
    return re.sub(r"\s+","",norm(x).upper())

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
    m=re.match(r"^(20\d{2})\s*Batch$",norm(x),re.I)
    return m.group(1) if m else ""

@st.cache_data(show_spinner=False)
def load_authoritative_data():
    """Clean and organize the two supplied Excel workbooks before retrieval."""
    if not MINOR_FILE.exists() or not STRUCT_FILE.exists():
        raise FileNotFoundError("The two authoritative Excel source files are missing from the data folder.")

    # ========================================================
    # 1) MINOR WORKBOOK — header-driven parsing
    # ========================================================
    minors=[]; placeholders=[]
    wb=openpyxl.load_workbook(MINOR_FILE,data_only=True,read_only=True)
    for ws in wb.worksheets:
        vals=list(ws.iter_rows(values_only=True))
        header_idx=None
        for i,row in enumerate(vals[:10]):
            hs=[norm(x).lower() for x in row]
            if "course code" in hs and any("course title" in x for x in hs):
                header_idx=i; break
        if header_idx is None: continue
        hs=[norm(x).lower() for x in vals[header_idx]]
        def find_col(*names):
            for name in names:
                for j,h in enumerate(hs):
                    if h==name.lower(): return j
            for j,h in enumerate(hs):
                if any(name.lower() in h for name in names): return j
            return None
        c={"code":find_col("course code"),"title":find_col("course title"),"l":find_col("l"),"t":find_col("t"),"p":find_col("p"),"credit":find_col("credit"),"semester":find_col("semester"),"pre":find_col("pre-rq","pre-req","prerequisite")}
        batch=""
        for rnum,row in enumerate(vals,start=1):
            for j in range(min(2,len(row))):
                b=batch_marker(row[j])
                if b: batch=b
            if rnum<=header_idx+1: continue
            code_raw=norm(row[c["code"]]) if c["code"] is not None else ""
            title_raw=norm(row[c["title"]]) if c["title"] is not None else ""
            credit=row[c["credit"]] if c["credit"] is not None else None
            sem=clean_semester(row[c["semester"]]) if c["semester"] is not None else ""
            pre_raw=norm(row[c["pre"]]) if c["pre"] is not None else ""
            if title_raw and title_raw.upper() in {"TBD","DON'T KNOW"} and not code_raw:
                placeholders.append({"source_workbook":MINOR_FILE.name,"source_sheet":ws.title,"source_row":rnum,"batch":batch,"placeholder_type":title_raw.upper(),"raw_title":title_raw,"raw_credits":credit,"raw_prerequisite":pre_raw})
                continue
            if not code_raw and not title_raw: continue
            if title_raw.lower() in {"finance minor","marketing minor","start -up minor","start-up minor"}: continue
            cu=code_raw.upper()
            if cu in {"DON'T KNOW","DONT KNOW"}: code_status="not_recorded_in_source"; code=""
            elif cu=="NEW": code_status="new_code_pending"; code=""
            elif code_raw: code_status="recorded"; code=norm_code(code_raw)
            else: code_status="missing_in_source"; code=""
            pre,pre_status=clean_prerequisite(pre_raw)
            minors.append({"source_workbook":MINOR_FILE.name,"source_sheet":ws.title,"source_row":rnum,"minor":ws.title,"batch":batch,"course_code_raw":code_raw,"course_code":code,"course_code_status":code_status,"course_title_raw":title_raw,"course_title":title_raw,"lecture_hours":row[c["l"]] if c["l"] is not None else None,"tutorial_hours":row[c["t"]] if c["t"] is not None else None,"practical_hours":row[c["p"]] if c["p"] is not None else None,"credits":credit,"semester":sem,"prerequisite_raw":pre_raw,"prerequisite":pre,"prerequisite_status":pre_status})

    # ========================================================
    # 2) SEMESTER-SPREAD WORKBOOK — block-driven parsing
    # ========================================================
    semesters=[]; structures=[]; requirements=[]
    wb=openpyxl.load_workbook(STRUCT_FILE,data_only=True,read_only=True)
    for ws in [wb[n] for n in wb.sheetnames[:5]]:
        vals=list(ws.iter_rows(values_only=True))
        blocks=[]
        for c0 in range(len(vals[1])):
            h2=norm(vals[1][c0]).lower()
            if h2 in {"course code","code"}:
                label=""
                for cc0 in [c0,c0+1,c0-1]:
                    if 0<=cc0<len(vals[0]):
                        v=norm(vals[0][cc0])
                        if re.fullmatch(r"S[1-8]",v,re.I): label=v.upper(); break
                if label: blocks.append((c0,label))
        batch=re.search(r"(20\d{2})",ws.title)
        batch=batch.group(1) if batch else ""
        basket=category=""
        for rnum,row in enumerate(vals[2:],start=3):
            b=norm(row[0]) if len(row)>0 else ""; cat=norm(row[1]) if len(row)>1 else ""
            if b: basket=b
            if cat: category=cat
            for c0,sem in blocks:
                vals7=list(row[c0:c0+7])+[None]*7
                code_raw=norm(vals7[0]); title_raw=norm(vals7[1]); pre_raw=norm(vals7[2])
                # Some source blocks (notably parts of Sem_Spread_2023)
                # contain a course title but no course-code header/value.
                # Preserve those rows as source records rather than dropping
                # them or inventing a code.
                if not code_raw and not title_raw: continue
                cu=code_raw.upper()
                if not code_raw: code_status="missing_in_source"; code=""
                elif cu in {"TBA","TBD"}: code_status="pending"; code=""
                elif cu in {"DON'T KNOW","DONT KNOW"}: code_status="not_recorded_in_source"; code=""
                else: code_status="recorded"; code=norm_code(code_raw)
                pre,pre_status=clean_prerequisite(pre_raw)
                semesters.append({"source_workbook":STRUCT_FILE.name,"source_sheet":ws.title,"source_row":rnum,"batch":batch,"basket":basket,"category":category,"semester":sem,"course_code_raw":code_raw,"course_code":code,"course_code_status":code_status,"course_title_raw":title_raw,"course_title":title_raw,"prerequisite_raw":pre_raw,"prerequisite":pre,"prerequisite_status":pre_status,"lecture_hours":vals7[3],"tutorial_hours":vals7[4],"practical_hours":vals7[5],"credits":vals7[6]})

    # ========================================================
    # 3) STRUCTURE SHEETS — exact column layouts, no invented codes
    # ========================================================
    for ws in [wb[n] for n in wb.sheetnames[5:]]:
        vals=list(ws.iter_rows(values_only=True))
        if ws.title in {"Struct_2022","Struct_2023"}:
            groups=[(5,6,"University Core"),(9,10,"Foundation"),(13,14,"Program Core"),(17,18,"Program Honors"),(21,22,"Honors Track 1"),(25,26,"Honors Track 2"),(29,30,"Internship/Capstone")]
            basket_cols=(1,2)
        else:
            groups=[(3,4,"University Core"),(7,8,"Program Core"),(11,12,"Electives")]
            basket_cols=(14,15)
        year=re.search(r"(20\d{2})",ws.title); year=year.group(1) if year else ""
        for rnum,row in enumerate(vals[2:],start=3):
            for title_col,credit_col,category in groups:
                ti,ci=title_col-1,credit_col-1
                title=norm(row[ti]) if ti<len(row) else ""; credit=row[ci] if ci<len(row) else None
                if not title or title.lower() in {"total credits","courses","course"}: continue
                structures.append({"source_workbook":STRUCT_FILE.name,"source_sheet":ws.title,"source_row":rnum,"structure_version":year,"category":category,"course_title_raw":title,"course_title":title,"credits":credit})
            bi,ci=basket_cols[0]-1,basket_cols[1]-1
            basket=norm(row[bi]) if bi<len(row) else ""; credit=row[ci] if ci<len(row) else None
            if basket and basket.lower() not in {"baskets","components"} and credit not in (None,""):
                requirements.append({"source_workbook":STRUCT_FILE.name,"source_sheet":ws.title,"source_row":rnum,"structure_version":year,"basket":basket,"credits":credit})

    minors_df=pd.DataFrame(minors); semesters_df=pd.DataFrame(semesters); structures_df=pd.DataFrame(structures); requirements_df=pd.DataFrame(requirements); placeholders_df=pd.DataFrame(placeholders)
    course_parts=[]
    for _,r in semesters_df[semesters_df.course_code!=""].iterrows():
        course_parts.append({"course_code":r.course_code,"course_title":r.course_title,"credits":r.credits,"prerequisite":r.prerequisite,"prerequisite_status":r.prerequisite_status,"batch":r.batch,"source_type":"semester_excel","source_sheet":r.source_sheet,"source_row":r.source_row})
    for _,r in minors_df[minors_df.course_code!=""].iterrows():
        course_parts.append({"course_code":r.course_code,"course_title":r.course_title,"credits":r.credits,"prerequisite":r.prerequisite,"prerequisite_status":r.prerequisite_status,"batch":r.batch,"source_type":"minor_excel","source_sheet":r.source_sheet,"source_row":r.source_row})
    courses=pd.DataFrame(course_parts)
    return minors_df,semesters_df,structures_df,requirements_df,courses,placeholders_df
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

@st.cache_resource(show_spinner="Preparing academic information…")
def build_rag():
    minors, semesters, structures, requirements, courses, placeholders = load_authoritative_data()
    records=[]
    for _,r in minors.iterrows():
        records.append(record_to_text(r.to_dict()))
    for _,r in semesters.iterrows():
        records.append(record_to_text(r.to_dict()))
    for _,r in structures.iterrows():
        records.append(record_to_text(r.to_dict()))

    chunks=[]
    for i,t in enumerate(records):
        chunks.extend(chunk_text(t,{"record_id":i,"source":"authoritative_excel"}))

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
                "minors":minors,"semesters":semesters,"structures":structures,"requirements":requirements,"courses":courses,"placeholders":placeholders}
    except Exception as e:
        # Explicitly labelled fallback so the system never pretends TF-IDF is dense RAG.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
        X=vec.fit_transform(texts)
        return {"mode":"lexical_fallback","chunks":chunks,"texts":texts,"vectorizer":vec,"X":X,
                "error":str(e),"minors":minors,"semesters":semesters,"structures":structures,"requirements":requirements,"courses":courses,"placeholders":placeholders}

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
    exact_code=courses[courses.course_code.eq(qcode)]
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

def _course_rows(courses, code):
    return courses[courses.course_code.eq(norm_code(code))].drop_duplicates()

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
        return {"answer":f"{code} — {title} is listed for: **{', '.join(sems)}** in the supplied semester-spread workbook. This is a source listing, not a guarantee of future registration availability.","status":"verified","evidence":evidence}
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
        choices=", ".join(f"{norm(r.course_code)} — {norm(r.course_title)}" for _,r in rows.iterrows())
        return {"answer":f"I found more than one match: **{choices}**. Which one do you mean?","status":"follow_up","evidence":_source_evidence(rows)}
    if kind=="none":
        return {"answer":f"I could not match **{target}** to a course in the supplied academic records. Please give the course code or full course name.","status":"follow_up","evidence":[]}

    target_code=norm_code(course.course_code); target_title=norm(course.course_title)
    target_rows=_course_rows(courses,target_code)
    if mode=="forward":
        # Forward lookup: what is required before taking the target course?
        prereq_values=[]; ev=[]
        for _,r in target_rows.iterrows():
            p=norm(r.get("prerequisite",""))
            if p and p not in prereq_values: prereq_values.append(p)
        ev=_source_evidence(target_rows)
        if not prereq_values or all(p.upper() in {"NIL","NONE","NA","N/A","-"} for p in prereq_values):
            return {"answer":f"**{target_code} — {target_title}** has no prerequisite recorded in the supplied academic data.","status":"verified","evidence":ev}
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
        return {"answer":f"I could not find any supplied course record that lists **{target_code} — {target_title}** as a prerequisite.","status":"verified","evidence":target_rows.head(8).to_dict("records") and _source_evidence(target_rows) or []}
    depdf=pd.DataFrame(dependents).drop_duplicates(subset=["course_code","course_title"])
    items=[]
    ev=[]
    for _,r in depdf.iterrows():
        items.append(f"**{norm(r.course_code)} — {norm(r.course_title)}**")
        ev.append({"source":r.get("source_sheet",""),"row":r.get("source_row",""),"workbook":r.get("source_workbook","Semester & Structure workbook"),"text":f"{norm(r.course_code)} — {norm(r.course_title)}; prerequisite: {norm(r.prerequisite)}"})
    verb="is" if len(items)==1 else "are"
    return {"answer":f"**{target_code} — {target_title}** {verb} listed as a prerequisite for:\n\n"+"\n".join(f"• {x}" for x in items),"status":"verified","evidence":ev}

def database_query_answer(query, minors, semesters, structures, courses):
    """Answer questions whose facts can be obtained directly from the two Excel workbooks.
    This is deliberately deterministic; the LLM only verbalizes grounded evidence later.
    Returns None when the query needs course/entity resolution or student logic.
    """
    ql=query.lower()
    # ----- minor database -----
    minor=_minor_match(ql,minors)
    if minor:
        rows=minors[minors.minor.astype(str).str.lower().eq(minor.lower())].copy()
        # Optional semester/batch filters.
        sem=_semester_number(ql)
        if "semester" in ql and any(x in ql for x in ["specified semester", "particular semester", "a semester"]) and not sem:
            return {"answer":f"Which semester should I use for the {minor} minor? Please specify a semester such as Semester 3 or Semester 5.","status":"follow_up","evidence":[]}
        if sem and "semester" in rows.columns:
            rows=rows[rows.semester.astype(str).str.upper().str.replace(" ","").eq(sem)]
        batch=re.search(r"\b(?:batch|year)\s*(2022|2023|2024|2025|2026)\b",ql)
        if batch and "batch" in rows.columns:
            b=batch.group(1); rows=rows[rows.batch.astype(str).str.contains(b,case=False,na=False)]
        rows=rows.drop_duplicates(subset=[c for c in ["course_code","course_title"] if c in rows.columns])
        if len(rows)==0:
            return {"answer":f"No matching {minor} minor course records were found for the specified filters in the supplied Excel data.","status":"insufficient","evidence":[]}
        if _wants_list(ql) or "minor" in ql:
            if any(x in ql for x in ["lecture", "tutorial", "practical", "l t p", "hours"]):
                fields=[("course_code","Code"),("course_title","Course"),("lecture_hours","L"),("tutorial_hours","T"),("practical_hours","P"),("credits","Credits"),("semester","Semester"),("prerequisite","Prerequisite")]
            else:
                fields=[("course_code","Code"),("course_title","Course"),("credits","Credits"),("semester","Semester"),("prerequisite","Prerequisite")]
            items=_format_rows(rows,fields)
            return {"answer":f"The supplied **{minor}** minor workbook records {len(rows)} matching course entries:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":_source_evidence(rows)}

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

def direct_answer(query, student_id, rag):
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
    routed=database_query_answer(q,minors,semesters,structures,courses)
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
            items=[f"{norm(r.course_code)} — {norm(r.course_title)} ({norm(r.credits)} credits; Semester {norm(r.semester)})" for _,r in rows.iterrows()]
            evidence=_source_evidence(rows)
            return {"answer":f"The supplied **{minor_name}** minor data contains {len(items)} distinct course records:\n\n"+"\n".join(f"- {x}" for x in items),"status":"verified","evidence":evidence}

    semno=_semester_number(ql)
    if semno and _is_list_query(ql):
        rows=semesters[semesters.semester.astype(str).str.upper().eq(semno)].drop_duplicates(subset=["course_code","course_title"])
        if len(rows):
            items=[f"{norm(r.course_code)} — {norm(r.course_title)} ({norm(r.credits)} credits)" for _,r in rows.iterrows()]
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
        choices=", ".join(sorted(f"{norm(r.course_code)} — {norm(r.course_title)}" for _,r in rows.iterrows()))
        return {"answer":f"I found multiple possible matches: **{choices}**. Please tell me which course you mean. I will not guess.","status":"follow_up","evidence":_source_evidence(rows)}
    if kind=="none":
        if any(x in ql for x in ["can i", "can ", "eligible", "register", "enroll", "enrol", "am i allowed"]):
            return {"answer":"To check eligibility, please provide/select the course you mean. If you are asking about your own eligibility, I also need the relevant student profile or synthetic student ID.","status":"follow_up","evidence":[]}
        if "credit" in ql and any(x in ql for x in ["degree","graduat","required","need"]):
            cats=structures.groupby(["source_sheet","category"],dropna=False).size().reset_index()
            return {"answer":"I need the applicable academic structure/batch to determine a graduation-credit requirement because the supplied workbook contains multiple structure years. Please provide the structure/year (for example, 2024, 2025, or 2026).","status":"follow_up","evidence":_source_evidence(cats)}
        if any(x in ql for x in ["can i","eligible","register","enroll","enrol"]):
            return {"answer":"I could not identify the course in the supplied academic data. Please provide the course name or code exactly as it appears in the university records.","status":"follow_up","evidence":[]}
        return {"answer":"I could not identify the requested information in the supplied university Excel sources. Please provide a more specific course name, course code, minor, semester, structure/year, or other detail.","status":"insufficient","evidence":[]}

    code=norm_code(course.course_code); title=norm(course.course_title)
    fact=_course_fact_answer(code,title,courses,semesters,ql)
    conflicts=material_conflicts(code,semesters)
    if conflicts:
        base_ev=_source_evidence(_course_rows(courses,code))
        return {"answer":f"The supplied semester-spread data contains differing prerequisite records for **{code} — {title}**. I will not silently choose between them. The source records should be verified against the current official academic authority.","status":"conflict","evidence":base_ev}
    if fact:
        # Student-specific wording still requires a profile.
        if any(x in ql for x in ["can i","can ","eligible","register","enroll","enrol","am i allowed"]):
            pass
        else: return fact

    # 3) Student eligibility.
    if any(x in ql for x in ["can i","can we","can ","eligible","register","enroll","enrol","am i allowed"]):
        if not student_id:
            return {"answer":f"I found **{code} — {title}**. To check your eligibility, please select a synthetic student profile so I can compare its recorded history with the source prerequisite.","status":"follow_up","evidence":_source_evidence(_course_rows(courses,code))}
        _,history=synthetic_students(); result=student_eligibility(student_id,course.to_dict(),history)
        ev=_source_evidence(_course_rows(courses,code))
        if result["status"]=="eligible_on_prerequisite_check":
            ans=f"For **{student_id}**, the recorded prerequisite evidence for {code} is satisfied. This verifies the prerequisite check only; the supplied sources do not establish every possible registration rule."
        elif result["status"]=="not_eligible_based_on_record":
            ans=f"For **{student_id}**, the recorded history does not satisfy the prerequisite check for {code}: a required prerequisite is recorded as failed. The supplied sources do not establish an override process."
        else:
            ans=f"I do not have enough student-record information to verify eligibility for {code}. Required prerequisite evidence not demonstrated: {', '.join(result['missing'])}."
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

SOURCE_NAMES = {
    "Minor_Courses_for_BTech_Students(3)": "Minor Courses workbook",
    "Semester_Spread_Structures_Sept_2026(3)": "Semester & Structure workbook",
}

def friendly_source(e):
    raw = str(e.get("source", "University Excel source"))
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


def gemini_understand_query(query, student_id=None):
    """Use Gemini before retrieval to interpret natural-language intent and entities.
    The returned retrieval query is only a routing aid; academic facts still come
    exclusively from the deterministic Excel/RAG verification layer.
    """
    key=st.secrets.get("GEMINI_API_KEY",os.getenv("GEMINI_API_KEY",""))
    if not key:
        return {"query_for_retrieval": query, "intent": "unknown", "course_terms": [], "needs_student_profile": False}, "GEMINI_API_KEY is not configured in Streamlit Secrets."

    prompt=f"""You are AIRA's query-understanding layer.
Convert the user's natural-language academic question into a precise retrieval intent.
Do NOT answer the question and do NOT invent any university facts.
Only identify what the user is asking and preserve course names/codes exactly as written.

Return ONLY valid JSON with these keys:
{{
  "query_for_retrieval": "short precise search query",
  "intent": "conversation | prerequisite | eligibility | dependent_courses | course_details | offering | minor | credits | graduation_requirement | ambiguous | insufficient | general_academic",
  "course_terms": ["course names or codes explicitly mentioned"],
  "needs_student_profile": true/false
}}

USER QUESTION:
{query}
STUDENT PROFILE SELECTED: {student_id or 'none'}
"""
    try:
        from google import genai
        import json
        client=genai.Client(api_key=key)
        models=[st.secrets.get("GEMINI_MODEL","gemini-2.5-flash"),"gemini-2.5-flash-lite"]
        last=None
        for model in dict.fromkeys(models):
            try:
                r=client.models.generate_content(model=model,contents=prompt,config={"response_mime_type":"application/json"})
                txt=getattr(r,"text",None)
                if txt:
                    obj=json.loads(txt)
                    if isinstance(obj,dict) and obj.get("query_for_retrieval"):
                        obj.setdefault("intent","unknown")
                        obj.setdefault("course_terms",[])
                        obj.setdefault("needs_student_profile",False)
                        return obj,None
            except Exception as e:
                last=e
        return {"query_for_retrieval":query,"intent":"unknown","course_terms":[],"needs_student_profile":False}, f"Gemini intent request failed: {last}"
    except Exception as e:
        return {"query_for_retrieval":query,"intent":"unknown","course_terms":[],"needs_student_profile":False}, f"Gemini intent SDK error: {e}"

def gemini_production_format(query, base, student_id=None):
    """Use Gemini in production, but only as a grounded interpretation/formatting layer.
    Academic facts come exclusively from the deterministic verification result and
    attached source evidence. Gemini is never allowed to add facts, courses, rules,
    prerequisites, credits, or recommendations.
    """
    key=st.secrets.get("GEMINI_API_KEY",os.getenv("GEMINI_API_KEY",""))
    if not key:
        return None, "GEMINI_API_KEY is not configured in Streamlit Secrets."

    status=base.get("status","verified")
    verified=base.get("answer","")
    evidence=base.get("evidence",[])
    evidence_text="\n".join(
        f"- {e.get('text','')} | workbook={e.get('workbook',e.get('source',''))} | sheet={e.get('source','')} | row={e.get('row','')}"
        for e in evidence
    )
    prompt=f"""You are AIRA's production response formatter.

The user's question has already been checked against the university's supplied Excel records.
Your job is to understand the user's intent and present ONLY the verified result below.

NON-NEGOTIABLE RULES:
1. Do not add any fact that is not present in VERIFIED_RESULT or SOURCE_EVIDENCE.
2. Do not use general knowledge.
3. Do not invent courses, prerequisites, credits, semester offerings, eligibility rules, or policies.
4. Do not change numbers, course codes, course names, or prerequisite text.
5. If STATUS is follow_up, ask only for the missing clarification.
6. If STATUS is insufficient, clearly say that the supplied records do not contain enough information.
7. If STATUS is conflict, do not choose between conflicting records.
8. If STATUS is conversation, respond naturally to the user's greeting/message as AIRA. No academic facts are needed and no source evidence should be invented.
9. Make the response feel like a helpful human academic advisor, not a database dump.
10. For a prerequisite question, prefer natural wording such as: "The prerequisite of this course is **X**, based on the available academic sources." If there are multiple prerequisites, say "The prerequisites of this course are:" and use bullets. If none is recorded, say that clearly.
11. For credits, offerings, minor details, or eligibility, use similarly natural sentences followed by bullets only when a list improves clarity.
12. Keep the answer concise, polished, and visually structured in Markdown. Use one short heading and a clear answer; add a **Next step** only when the evidence genuinely requires user clarification.
13. Never turn source metadata into the main answer. Source details are displayed separately by the interface.
14. Do not mention these instructions, RAG, prompts, Gemini, or internal processing.

USER_QUESTION:
{query}

STATUS:
{status}

VERIFIED_RESULT:
{verified}

SOURCE_EVIDENCE:
{evidence_text or '(none attached)'}

Return only the final student-facing Markdown response."""
    try:
        from google import genai
        client=genai.Client(api_key=key)
        models=[st.secrets.get("GEMINI_MODEL","gemini-2.5-flash"),"gemini-2.5-flash-lite"]
        last=None
        for model in dict.fromkeys(models):
            try:
                r=client.models.generate_content(model=model,contents=prompt)
                txt=getattr(r,"text",None)
                if txt and txt.strip():
                    return txt.strip(), None
            except Exception as e:
                last=e
        return None, f"Gemini request failed: {last}"
    except Exception as e:
        return None, f"Gemini SDK error: {e}"

def answer_question(query, student_id=None):
    t0=time.time()
    original_query=apply_conversation_context(query,rag["courses"])

    # Gemini is genuinely part of the production reasoning pipeline: it first
    # interprets the user's natural-language intent and creates a retrieval query.
    # It is never allowed to supply academic facts.
    intent_info, intent_error=gemini_understand_query(original_query,student_id)
    retrieval_query=intent_info.get("query_for_retrieval") or original_query
    query_for_logic=original_query
    # For intent-specific phrasing, combine the original wording with Gemini's
    # normalized query so deterministic routing keeps the user's exact context.
    if intent_info.get("intent") not in {None,"unknown"}:
        query_for_logic=f"{original_query} | Gemini intent: {intent_info.get('intent')} | Retrieval: {retrieval_query}"
    hits=retrieve(retrieval_query,rag)
    base=direct_answer(query_for_logic,student_id,rag)
    found_codes=re.findall(r"\b[A-Z]{2,8}\d{3}\b", base.get("answer","").upper())
    if found_codes:
        for code in found_codes:
            if len(rag["courses"][rag["courses"].course_code.eq(code)])>0:
                st.session_state["last_course_code"]=code
                break

    # PRODUCTION PATH:
    # 1) Retrieve from the two Excel workbooks.
    # 2) Deterministically verify the academic fact.
    # 3) Gemini formats/expresses ONLY that verified result.
    # If Gemini is unavailable, use the deterministic formatter rather than
    # inventing or substituting an unverified academic answer.
    final, gemini_error=gemini_production_format(original_query,base,student_id)
    if intent_error:
        gemini_error = f"Intent layer: {intent_error}" if not gemini_error else f"Intent layer: {intent_error} | Response layer: {gemini_error}"
    if not final:
        final=format_production_answer(base)
    return final, base, hits, time.time()-t0, gemini_error

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


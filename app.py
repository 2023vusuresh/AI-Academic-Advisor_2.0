import os,re,json,html,hashlib
from pathlib import Path
from typing import TypedDict,Any
import pandas as pd
import numpy as np
import streamlit as st

BASE=Path(__file__).resolve().parent
SRC=BASE/'source_data'
SYN=BASE/'synthetic_data'
SYN.mkdir(exist_ok=True)

# Synthetic data is explicitly separated from university sources.
STUDENTS=pd.DataFrame([
 {'student_id':'SYN001','label':'Synthetic Student 1','batch':'2026','programme':'BTech','current_semester':5,'minor':'Finance'},
 {'student_id':'SYN002','label':'Synthetic Student 2','batch':'2026','programme':'BTech','current_semester':6,'minor':'Psychology'},
])
STUDENTS.to_csv(SYN/'synthetic_students.csv',index=False)


def norm(x):
 x=str(x or '').lower().replace('&',' and ')
 x=re.sub(r'[^a-z0-9]+',' ',x)
 return re.sub(r'\s+',' ',x).strip()

def clean(x): return re.sub(r'\s+',' ',str(x or '')).strip()

def source_label(path): return path.name

@st.cache_data(show_spinner=False)
def load_sources():
 records=[]; structures=[]; minors=[]; docs=[]
 import openpyxl
 wb=openpyxl.load_workbook(SRC/'Semester_Spread_Structures_Sept_2026.xlsx',data_only=True,read_only=True)
 for ws in wb.worksheets:
  vals=list(ws.iter_rows(values_only=True))
  if ws.title.startswith('Sem Spread'):
   if len(vals)<2: continue
   h2=list(vals[1]); h1=list(vals[0])
   starts=[]
   for i,v in enumerate(h2):
    if norm(v)=='course code':
     sem=''
     for j in range(i,-1,-1):
      if h1[j] is not None:
       sem=clean(h1[j]); break
     starts.append((i,sem))
   for rno,row in enumerate(vals[2:],start=3):
    basket=clean(row[0]) if len(row)>0 else ''
    basket_min=clean(row[3]) if len(row)>3 else ''
    for i,sem in starts:
     if i+6>=len(row): continue
     code=clean(row[i]); title=clean(row[i+1]); pre=clean(row[i+2]); cred=row[i+6]
     if not code or not title: continue
     if norm(code) in {'course code','code'}: continue
     rec={'source':'Semester_Spread_Structures_Sept_2026.xlsx','sheet':ws.title,'row':rno,'semester':sem,'course_code':code,'course_title':title,'prerequisite':pre,'credits':cred,'basket':basket,'fixed_min':basket_min}
     records.append(rec)
     docs.append({**rec,'text':f"{ws.title} row {rno}: {code} | {title} | prerequisite: {pre} | credits: {cred} | semester: {sem} | basket: {basket}"})
  elif ws.title.startswith('Struct_'):
   for rno,row in enumerate(vals,start=1):
    txt=' | '.join(clean(x) for x in row if x is not None)
    if txt: structures.append({'source':'Semester_Spread_Structures_Sept_2026.xlsx','sheet':ws.title,'row':rno,'text':txt})
    if txt: docs.append({'source':'Semester_Spread_Structures_Sept_2026.xlsx','sheet':ws.title,'row':rno,'text':txt})
 wb.close()

 wb2=openpyxl.load_workbook(SRC/'Minor_Courses_for_BTech_Students.xlsx',data_only=True,read_only=True)
 for ws in wb2.worksheets:
  vals=list(ws.iter_rows(values_only=True)); batch=''
  for rno,row in enumerate(vals,start=1):
   row=list(row)
   if row and isinstance(row[0],str) and 'batch' in row[0].lower(): batch=clean(row[0])
   # Identify course-code/title/credit columns from actual workbook headers.
   code=title=pre=None; credit=sem=None
   for i,v in enumerate(row):
    nv=norm(v)
    if nv=='course code': code_i=i
    if nv in {'course title','course title '} : title_i=i
    if nv in {'credit','credits'}: credit_i=i
    if nv in {'semester','sem'}: sem_i=i
    if nv in {'pre rq','pre req','pre requisite','pre req'}: pre_i=i
   try: code=clean(row[code_i])
   except: code=''
   try: title=clean(row[title_i])
   except: title=''
   try: credit=row[credit_i]
   except: credit=None
   try: sem=row[sem_i]
   except: sem=None
   try: pre=clean(row[pre_i])
   except: pre=''
   if title and code and norm(code) not in {'course code','dont know'}:
    rec={'source':'Minor_Courses_for_BTech_Students.xlsx','sheet':ws.title,'row':rno,'batch':batch,'course_code':code,'course_title':title,'credits':credit,'semester':sem,'prerequisite':pre}
    minors.append(rec); docs.append({**rec,'text':f"{ws.title} | {batch} | {code} | {title} | credits: {credit} | semester: {sem} | prerequisite: {pre}"})
 wb2.close()

 # Raw PDFs: page-level source documents.
 from pypdf import PdfReader
 for fn in ['Student_Handbook_Aug_2026.pdf','SOP_STUDENT_17082026_Final.pdf']:
  reader=PdfReader(SRC/fn)
  for p,page in enumerate(reader.pages,1):
   txt=clean(page.extract_text() or '')
   if txt: docs.append({'source':fn,'sheet':f'page {p}','row':p,'text':txt})

 courses=pd.DataFrame(records+minors)
 if not courses.empty:
  courses['credits_num']=pd.to_numeric(courses['credits'],errors='coerce')
 return courses,pd.DataFrame(structures),pd.DataFrame(minors),docs

courses,structures,minors,raw_docs=load_sources()

# Build compact text index from ORIGINAL source pages/rows.
@st.cache_resource(show_spinner=False)
def build_index(docs):
 texts=[d['text'] for d in docs]
 vec=None; X=None; embedder=None; E=None
 try:
  from sklearn.feature_extraction.text import TfidfVectorizer
  vec=TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=1); X=vec.fit_transform(texts)
 except Exception: pass
 try:
  from sentence_transformers import SentenceTransformer
  embedder=SentenceTransformer('all-MiniLM-L6-v2'); E=embedder.encode(texts,normalize_embeddings=True,show_progress_bar=False)
 except Exception: pass
 return vec,X,embedder,E
VEC,TFIDF,EMBEDDER,EMB=build_index(raw_docs)

def entity_matches(q):
 nq=norm(q)
 if courses.empty: return pd.DataFrame()
 c=courses.copy(); c['_code']=c.course_code.map(norm); c['_title']=c.course_title.map(norm)
 # exact code/title first, then token containment; preserve source rows.
 exact=c[(c['_code']==nq)|(c['_title']==nq)]
 if not exact.empty: return exact
 # Common natural-language wrapper removal.
 q2=re.sub(r'\b(how many|credits?|is|are|the|for|course|subject|have|has|does|do|what|which|semester|offered|offer|prerequisite|pre requisite)\b',' ',nq)
 q2=re.sub(r'\s+',' ',q2).strip()
 if not q2: return pd.DataFrame()
 mask=c['_title'].str.contains(re.escape(q2),na=False)|c['_code'].str.contains(re.escape(q2),na=False)
 return c[mask]

def structured_answer(q):
 nq=norm(q)
 # Course credit/attribute questions.
 wants_credit=bool(re.search(r'\b(credit|credits)\b',nq))
 wants_pre=bool(re.search(r'\b(prerequisite|pre requisite|pre req)\b',nq))
 wants_sem=bool(re.search(r'\b(semester|offered|offer)\b',nq))
 if wants_credit or wants_pre or wants_sem:
  # remove question intent words and academic filler
  candidate=re.sub(r'\b(how many|credits?|what|is|are|the|for|course|subject|does|do|have|has|which|semester|offered|offer|prerequisite|pre requisite|pre req)\b',' ',nq)
  candidate=re.sub(r'\s+',' ',candidate).strip()
  m=entity_matches(candidate)
  if m.empty:
   # Try whole question for exact title/code
   m=entity_matches(nq)
  if m.empty: return None
  # Multiple distinct values = conflict/ambiguity.
  if wants_credit:
   vals=sorted({str(x) for x in m.credits_num.dropna().tolist()})
   if len(vals)==1: return f"**{m.iloc[0].course_title}** has **{vals[0]} credit(s)** in the supplied university source.\n\nSource: {m.iloc[0].source}, {m.iloc[0].sheet}, row {int(m.iloc[0].row)}."
   if len(vals)>1: return 'The supplied university sources contain different credit values for this course. I will not choose one without the applicable batch/programme context.'
  if wants_pre:
   vals=sorted({clean(x) for x in m.prerequisite if clean(x)})
   if len(vals)==1: return f"**{m.iloc[0].course_title}** has prerequisite **{vals[0]}** in the supplied university source.\n\nSource: {m.iloc[0].source}, {m.iloc[0].sheet}, row {int(m.iloc[0].row)}."
   if len(vals)>1: return 'The supplied sources show different prerequisites for this course. Please specify the batch/programme.'
  if wants_sem:
   vals=sorted({clean(x) for x in m.semester if clean(x)})
   if len(vals)==1: return f"**{m.iloc[0].course_title}** is listed for **{vals[0]}** in the supplied source.\n\nSource: {m.iloc[0].source}, {m.iloc[0].sheet}, row {int(m.iloc[0].row)}."
   if len(vals)>1: return 'This course appears in more than one semester across the supplied structures. Please specify the batch/structure.'

 # Count minor courses only when minor is explicitly named.
 if 'minor' in nq and any(x in nq for x in ['how many','count','number of']):
  minor_name=None
  for name in minors.sheet.unique() if not minors.empty else []:
   if norm(name) in nq: minor_name=name; break
  if minor_name:
   mm=minors[minors.sheet==minor_name]
   # Exclude header/dummy rows already excluded.
   return f"The supplied **{minor_name}** workbook contains **{len(mm)} course entries** across the batches represented in that sheet. Specify a batch if you need the count for one batch."

 # Explicit total-credit aggregation only with a named structure in the actual workbook.
 if 'total credits' in nq or ('credits' in nq and ('structure' in nq or 'programme structure' in nq)):
  matches=[s for s in structures.sheet.unique() if norm(s) in nq]
  if matches:
   return f"The source contains structure **{matches[0]}**. The detailed structure rows are available from the original Excel workbook; I will not infer a programme total unless the workbook explicitly provides it."
 return None

def rag(q,k=5):
 if not raw_docs: return []
 scores=np.zeros(len(raw_docs))
 if VEC is not None and TFIDF is not None:
  scores += np.asarray(VEC.transform([q]).dot(TFIDF.T).toarray()[0])
 if EMBEDDER is not None and EMB is not None:
  qe=EMBEDDER.encode([q],normalize_embeddings=True)[0]; scores += np.asarray(EMB@qe)*0.8
 idx=np.argsort(scores)[::-1][:k]
 return [(raw_docs[i],float(scores[i])) for i in idx if scores[i]>0]

def llm_answer(q,evidence):
 key=os.getenv('GEMINI_API_KEY')
 if not key: return None
 try:
  from langchain_google_genai import ChatGoogleGenerativeAI
  from langchain_core.prompts import ChatPromptTemplate
  model=ChatGoogleGenerativeAI(model='gemini-2.5-flash',google_api_key=key,temperature=0)
  prompt=ChatPromptTemplate.from_messages([('system','Answer only from the supplied evidence. If evidence does not establish the answer, say so. Do not invent facts. Be concise and cite source filename/page or sheet/row.'),('human','Question: {q}\nEvidence:\n{e}')])
  ev='\n\n'.join(f"SOURCE={d.get('source')} | LOCATION={d.get('sheet')} row/page={d.get('row')}\n{d.get('text')}" for d,s in evidence)
  return model.invoke(prompt.format_messages(q=q,e=ev)).content
 except Exception: return None

# LangGraph orchestration
try:
 from langgraph.graph import StateGraph,START,END
 class State(TypedDict, total=False): query:str; answer:str; evidence:list; route:str
 def plan(s):
  a=structured_answer(s['query'])
  return {'answer':a or '', 'route':'structured' if a else 'rag'}
 def retrieve(s): return {'evidence':rag(s['query'],5)}
 def generate(s):
  if s.get('answer'): return s
  ev=s.get('evidence',[]); ans=llm_answer(s['query'],ev)
  if not ans:
   if not ev: ans='I could not find this in the supplied university sources.'
   else:
    ans='I found relevant material in the supplied university sources, but it does not establish a precise answer to this question.\n\n' + '\n'.join(f"• {d['source']} — {d['sheet']}" for d,_ in ev[:3])
  return {'answer':ans}
 g=StateGraph(State); g.add_node('plan',plan); g.add_node('retrieve',retrieve); g.add_node('generate',generate); g.add_edge(START,'plan'); g.add_conditional_edges('plan',lambda s:s['route'],{'structured':'generate','rag':'retrieve'}); g.add_edge('retrieve','generate'); g.add_edge('generate',END); GRAPH=g.compile()
except Exception:
 GRAPH=None

def answer(q):
 if not clean(q): return 'Please enter an academic question.'
 if GRAPH:
  s=GRAPH.invoke({'query':q}); return s.get('answer','I could not establish an answer from the supplied sources.')
 a=structured_answer(q)
 return a or 'I could not establish an answer from the supplied sources.'

st.set_page_config(page_title='AIRA | Original University Sources',page_icon='🎓',layout='wide')
st.markdown('''<style>.block-container{max-width:1000px}.hero{padding:32px;border-radius:22px;background:#0b2d63;color:white;margin-bottom:18px}.hero h1{margin:0}.source{font-size:12px;color:#64748b}</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>AIRA</h1><p>Academic Intelligence & Registration Assistant</p><small>Source-first • Vidyashilp University documents only • No invented academic facts</small></div>',unsafe_allow_html=True)
with st.sidebar:
 st.header('Source status')
 st.write('Original PDFs:',len([x for x in raw_docs if x['source'].endswith('.pdf')]))
 st.write('Original Excel rows indexed:',len(courses))
 st.write('Synthetic students:',len(STUDENTS))
 st.caption('Synthetic students are not university records.')
 profile=st.selectbox('Synthetic profile',['New / General User']+STUDENTS.student_id.tolist())
q=st.chat_input('Ask about courses, credits, prerequisites, offerings, regulations…')
if q:
 st.chat_message('user').write(q)
 st.chat_message('assistant').markdown(answer(q))
else:
 st.markdown('Ask a question. Examples:')
 st.write('• How many credits does Financial and Management Accounting have?')
 st.write('• What is the prerequisite for Corporate Finance?')
 st.write('• Where is Vidyashilp University located?')
 st.write('• What does the Student Handbook say about [topic]?')

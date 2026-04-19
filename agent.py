"""
agent.py — MediCare Hospital Assistant
Agentic AI Capstone 2026 | Dr. Kanthi Kiran Sirra

This module builds and exports the compiled LangGraph app.
Import with: from agent import build_app
"""

import os
import uuid
from datetime import datetime
from typing import TypedDict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# Streamlit Cloud SQLite workaround for ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────

class CapstoneState(TypedDict):
    question:      str
    messages:      List[dict]
    route:         str
    retrieved:     str
    sources:       List[str]
    tool_result:   str
    answer:        str
    faithfulness:  float
    eval_retries:  int
    user_name:     Optional[str]


# ─────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {
        'id': 'doc_001',
        'topic': 'OPD Timings',
        'text': (
            'MediCare General Hospital OPD (Outpatient Department) operates Monday to Saturday. '
            'Morning session runs from 8:00 AM to 1:00 PM. Evening session runs from 4:00 PM to 8:00 PM. '
            'On Sundays and public holidays, only the Emergency Department is operational 24/7. '
            'Patients are advised to arrive 15 minutes before their slot for registration. '
            'Walk-in patients are accepted but appointment holders get priority. '
            'OPD registration counter opens at 7:45 AM on weekdays. '
            'The last token for morning OPD is issued at 12:30 PM and for evening OPD at 7:30 PM.'
        )
    },
    {
        'id': 'doc_002',
        'topic': 'Doctor Specialties and Departments',
        'text': (
            'MediCare General Hospital has the following specialist departments: '
            'Cardiology (Dr. Ramesh Nair — Mon, Wed, Fri), '
            'Orthopedics (Dr. Priya Sharma — Tue, Thu, Sat), '
            'Neurology (Dr. Anil Mehta — Mon, Thu), '
            'Pediatrics (Dr. Sunita Rao — all weekdays), '
            'Gynecology (Dr. Kavitha Reddy — all weekdays), '
            'General Medicine (Dr. Suresh Kumar — all weekdays), '
            'Dermatology (Dr. Meena Pillai — Tue, Fri), '
            'ENT (Dr. Rajiv Menon — Mon, Wed, Sat), '
            'Ophthalmology (Dr. Lakshmi Devi — Wed, Sat). '
            'Patients should consult the General Medicine department if unsure which specialist to see.'
        )
    },
    {
        'id': 'doc_003',
        'topic': 'Consultation and OPD Fees',
        'text': (
            'OPD consultation fees at MediCare General Hospital are as follows: '
            'General Medicine — ₹300 per consultation. '
            'Specialist consultation (Cardiology, Neurology, Orthopedics, Gynecology) — ₹500 per consultation. '
            'Pediatrics — ₹400 per consultation. '
            'Dermatology and ENT — ₹400 per consultation. '
            'Follow-up consultation within 7 days — 50% discount on consultation fee. '
            'Senior citizens (age 60+) receive a 20% discount on all consultation fees. '
            'Payment accepted by cash, UPI, debit card, and credit card at the billing counter. '
            'All fees must be paid at the billing counter before meeting the doctor.'
        )
    },
    {
        'id': 'doc_004',
        'topic': 'Appointment Booking',
        'text': (
            'Appointments at MediCare General Hospital can be booked in three ways: '
            '1. Online via the hospital website: www.medicare-hyd.in/appointments '
            '2. Phone: Call 040-6600-1234 between 8 AM and 8 PM on working days. '
            '3. Walk-in: Visit the OPD registration counter in person. '
            'For online booking, select your department, preferred doctor, and date. '
            'A confirmation SMS is sent to your registered mobile number within 15 minutes. '
            'Cancellations must be made at least 2 hours before the appointment. '
            "Repeat/follow-up appointments can be booked directly at the doctor's desk after consultation. "
            'Emergency cases do not require an appointment — proceed directly to the Emergency Department.'
        )
    },
    {
        'id': 'doc_005',
        'topic': 'Insurance and Cashless Treatment',
        'text': (
            'MediCare General Hospital is empanelled with over 25 insurance providers. '
            'Cashless treatment is available for: Star Health, United India, New India Assurance, '
            'HDFC ERGO, ICICI Lombard, Bajaj Allianz, Religare (Care Health), and Aditya Birla Health Insurance. '
            'For cashless admission, patients must present their insurance e-card and a valid photo ID at the insurance desk. '
            'The insurance desk is open Monday to Saturday, 9 AM to 5 PM. '
            'Pre-authorization from the insurer is required for planned surgeries and takes 4-24 hours. '
            'Government schemes: Aarogyasri (Telangana) and Ayushman Bharat (PM-JAY) are accepted. '
            'For reimbursement claims, all original bills and discharge summary must be collected at discharge. '
            'Contact the insurance helpdesk at insurance@medicare-hyd.in for queries.'
        )
    },
    {
        'id': 'doc_006',
        'topic': 'Emergency Department',
        'text': (
            'MediCare General Hospital Emergency Department is operational 24 hours a day, 7 days a week, 365 days a year. '
            'Emergency helpline number: 040-6600-9999 (toll-free, always active). '
            'Ambulance service: Call 104 or 040-6600-8888. '
            'The ED is equipped for cardiac emergencies, trauma, stroke, respiratory distress, and obstetric emergencies. '
            'A senior resident doctor and trained nursing staff are present at all times. '
            'Triage is performed immediately on arrival — critically ill patients are seen first. '
            'ICU (Intensive Care Unit) with 20 beds is adjacent to the Emergency Department. '
            'Do NOT use the OPD helpline for emergencies — always call 040-6600-9999 directly.'
        )
    },
    {
        'id': 'doc_007',
        'topic': 'Pharmacy',
        'text': (
            'MediCare General Hospital has an in-house pharmacy located on the Ground Floor, near the main exit. '
            'Pharmacy hours: Open 24 hours on all days including Sundays and holidays. '
            'All prescription medicines are dispensed against a valid doctor prescription only. '
            'Generic medicines are available at subsidised rates under the Pradhan Mantri Bhartiya Janaushadhi Pariyojana. '
            'Over-the-counter (OTC) medicines, first aid supplies, and health supplements are available without prescription. '
            'Home delivery of medicines is available within 10 km radius — call 040-6600-5678. '
            'Patients can request a printed prescription summary from the doctor which can be used at any external pharmacy. '
            'Pharmacy contact: pharmacy@medicare-hyd.in'
        )
    },
    {
        'id': 'doc_008',
        'topic': 'Diagnostic Lab and Radiology',
        'text': (
            'The Diagnostic Laboratory at MediCare General Hospital is on the First Floor. '
            'Lab timings: Monday to Saturday 7:00 AM to 7:00 PM. Sunday 8:00 AM to 12:00 PM. '
            'Tests offered: Complete Blood Count (CBC), Lipid Profile, Liver Function Test (LFT), '
            'Kidney Function Test (KFT), Blood Sugar (Fasting/PP/Random), Thyroid Profile (T3/T4/TSH), '
            'Urine Routine, Urine Culture, ECG, and Echocardiogram. '
            'Radiology services include: Digital X-Ray, Ultrasound, CT Scan, and MRI. '
            'Reports for routine blood tests are ready within 6 hours. '
            'Fasting required for Blood Sugar (Fasting) and Lipid Profile — no food 8-12 hours before the test. '
            'Home sample collection available — call 040-6600-7777 to schedule.'
        )
    },
    {
        'id': 'doc_009',
        'topic': 'Health Packages and Preventive Check-ups',
        'text': (
            'MediCare General Hospital offers the following preventive health packages: '
            'Basic Health Package — ₹999: CBC, Blood Sugar, Urine Routine, ECG, doctor consultation. '
            "Women's Wellness Package — ₹2499: All basic tests + Thyroid Profile, Pap Smear, Mammography, gynecology consultation. "
            'Cardiac Care Package — ₹3499: Lipid Profile, ECG, Echo, Stress Test, cardiology consultation. '
            'Senior Citizen Package — ₹1999: CBC, KFT, LFT, Blood Sugar, Thyroid, X-Ray chest, physician consultation. '
            'Diabetes Management Package — ₹1499: HbA1c, Fasting Sugar, Urine Microalbumin, KFT, eye check. '
            'Packages are available Monday to Saturday — prior booking required at 040-6600-1234. '
            'All packages include a printed health report and follow-up consultation within 30 days.'
        )
    },
    {
        'id': 'doc_010',
        'topic': 'Hospital Location, Parking, and General Information',
        'text': (
            'MediCare General Hospital is located at Plot No. 42, Banjara Hills Road No. 12, Hyderabad — 500034. '
            'Nearest metro station: Jubilee Hills Check Post (600 metres, 8-minute walk). '
            'Bus routes: TSRTC buses 5K, 10C, and 49F stop directly in front of the hospital. '
            'Free parking is available in the hospital basement for up to 2 hours. '
            'Extended parking: ₹50 per hour beyond the first 2 hours. '
            'Main helpline: 040-6600-1234. Emergency: 040-6600-9999. '
            'Email: info@medicare-hyd.in. Website: www.medicare-hyd.in. '
            'The hospital has 350 beds, 4 operation theatres, and a blood bank. '
            'Visiting hours for inpatients: 10 AM to 12 PM and 5 PM to 7 PM only.'
        )
    }
]


# ─────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """You are a routing assistant for a hospital chatbot.
Given a patient question, decide the best route.

Routes:
- retrieve: question needs information from the hospital knowledge base
  (OPD timings, doctors, fees, appointments, insurance, emergency, pharmacy, lab, packages, location)
- tool: question requires current date or time (e.g., 'Is OPD open now?', 'What time is it?', 'What day is today?')
- memory_only: simple greeting, thank you, or follow-up that needs no new information

Reply with ONE word only: retrieve, tool, or memory_only

Question: {question}"""

SYSTEM_PROMPT = """You are the MediCare General Hospital patient assistant.
You help patients with information about OPD timings, doctors, fees, appointments,
insurance, emergency services, pharmacy, lab tests, and health packages.

RULES:
1. Answer ONLY from the provided CONTEXT or TOOL RESULT. Do not use outside knowledge.
2. If the answer is not in the context, say clearly: "I don't have that information.
   Please call our helpline at 040-6600-1234."
3. NEVER give medical advice or diagnose conditions. Always redirect clinical questions to doctors.
4. For ANY emergency, immediately provide: Emergency number 040-6600-9999.
5. Be warm, empathetic, and professional at all times.
6. Address the patient by name if their name is known.
7. Keep answers concise and clear — patients are often anxious or in a hurry.

GROUNDING RULE: Your answer must be entirely based on the CONTEXT or TOOL RESULT below.
Do NOT add information that is not present in the context."""

EVAL_PROMPT = """Rate the faithfulness of the following answer to the given context on a scale from 0.0 to 1.0.
Faithfulness means the answer contains ONLY information that is present in the context — no hallucinations.

Context:
{context}

Answer:
{answer}

Reply with a single decimal number between 0.0 and 1.0 only. No explanation. Example: 0.85"""

MAX_EVAL_RETRIES = 2


# ─────────────────────────────────────────────────────────────────
# NODE FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def make_memory_node():
    def memory_node(state: CapstoneState) -> dict:
        messages  = state.get('messages', [])
        user_name = state.get('user_name', None)
        question  = state['question']
        messages  = messages + [{'role': 'user', 'content': question}]
        if len(messages) > 6:
            messages = messages[-6:]
        lower_q = question.lower()
        if 'my name is' in lower_q:
            parts = lower_q.split('my name is')
            if len(parts) > 1:
                name_raw = parts[1].strip().split()
                if name_raw:
                    user_name = name_raw[0].capitalize()
        return {'messages': messages, 'user_name': user_name}
    return memory_node


def make_router_node(llm):
    def router_node(state: CapstoneState) -> dict:
        prompt   = ROUTER_PROMPT.format(question=state['question'])
        response = llm.invoke(prompt)
        route    = response.content.strip().lower()
        if route not in ('retrieve', 'tool', 'memory_only'):
            route = 'retrieve'
        return {'route': route}
    return router_node


def make_retrieval_node(embedder, collection):
    def retrieval_node(state: CapstoneState) -> dict:
        question = state['question']
        q_emb    = embedder.encode([question]).tolist()
        results  = collection.query(query_embeddings=q_emb, n_results=3)
        docs     = results['documents'][0]
        metas    = results['metadatas'][0]
        sources  = [m['topic'] for m in metas]
        parts    = [f"[{m['topic']}]\n{d}" for d, m in zip(docs, metas)]
        retrieved = '\n\n'.join(parts)
        return {'retrieved': retrieved, 'sources': sources}
    return retrieval_node


def skip_retrieval_node(state: CapstoneState) -> dict:
    return {'retrieved': '', 'sources': []}


def tool_node(state: CapstoneState) -> dict:
    """DateTime tool — never raises exceptions."""
    try:
        now          = datetime.now()
        day_name     = now.strftime('%A')
        date_str     = now.strftime('%d %B %Y')
        time_str     = now.strftime('%I:%M %p')
        hour         = now.hour
        minute       = now.minute
        weekday      = now.weekday()
        is_weekday   = weekday < 6
        time_minutes = hour * 60 + minute
        morning_open = (8 * 60 <= time_minutes < 13 * 60)
        evening_open = (16 * 60 <= time_minutes < 20 * 60)
        opd_open     = is_weekday and (morning_open or evening_open)

        if day_name == 'Sunday':
            opd_status = 'OPD is CLOSED today (Sunday). Only Emergency is open 24/7.'
        elif opd_open:
            session    = 'Morning Session' if morning_open else 'Evening Session'
            opd_status = f'OPD is currently OPEN ({session}).'
        else:
            if time_minutes < 8 * 60:
                opd_status = 'OPD has not opened yet today. Morning session starts at 8:00 AM.'
            elif 13 * 60 <= time_minutes < 16 * 60:
                opd_status = 'OPD Morning Session has closed. Evening session opens at 4:00 PM.'
            else:
                opd_status = 'OPD is CLOSED for today. Reopens tomorrow at 8:00 AM.'

        result = (
            f'Current Date: {date_str}\n'
            f'Current Time: {time_str}\n'
            f'Day: {day_name}\n'
            f'OPD Status: {opd_status}'
        )
    except Exception as e:
        result = f'DateTime tool error: {str(e)}'

    return {'tool_result': result}


def make_answer_node(llm):
    def answer_node(state: CapstoneState) -> dict:
        question     = state['question']
        retrieved    = state.get('retrieved', '')
        tool_result  = state.get('tool_result', '')
        messages     = state.get('messages', [])
        user_name    = state.get('user_name', None)
        eval_retries = state.get('eval_retries', 0)

        context_block = ''
        if retrieved:
            context_block += f'CONTEXT FROM KNOWLEDGE BASE:\n{retrieved}\n\n'
        if tool_result:
            context_block += f'TOOL RESULT (DateTime):\n{tool_result}\n\n'
        if not context_block:
            context_block = 'No external context available. Use conversation history only.'

        history_str = ''
        for msg in messages[:-1]:
            role = 'Patient' if msg['role'] == 'user' else 'Assistant'
            history_str += f"{role}: {msg['content']}\n"

        retry_hint = ''
        if eval_retries > 0:
            retry_hint = '\n[NOTE: Previous answer was not well-grounded. Be more precise and stick strictly to the context.]'

        name_hint = f" The patient's name is {user_name}." if user_name else ''

        full_prompt = f"""{SYSTEM_PROMPT}{name_hint}{retry_hint}

CONVERSATION HISTORY:
{history_str}

{context_block}
Patient Question: {question}

Assistant Answer:"""

        response = llm.invoke(full_prompt)
        return {'answer': response.content.strip()}
    return answer_node


def make_eval_node(llm):
    def eval_node(state: CapstoneState) -> dict:
        retrieved    = state.get('retrieved', '')
        answer       = state.get('answer', '')
        eval_retries = state.get('eval_retries', 0)

        if not retrieved:
            return {'faithfulness': 1.0, 'eval_retries': eval_retries}

        prompt   = EVAL_PROMPT.format(context=retrieved[:1500], answer=answer)
        response = llm.invoke(prompt)
        raw      = response.content.strip()

        try:
            score = float(raw.split()[0])
            score = max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            score = 0.5

        new_retries = eval_retries + (1 if score < 0.7 else 0)
        return {'faithfulness': score, 'eval_retries': new_retries}
    return eval_node


def save_node(state: CapstoneState) -> dict:
    messages = state.get('messages', [])
    answer   = state.get('answer', '')
    messages = messages + [{'role': 'assistant', 'content': answer}]
    return {'messages': messages}


# ─────────────────────────────────────────────────────────────────
# CONDITIONAL EDGES
# ─────────────────────────────────────────────────────────────────

def route_decision(state: CapstoneState) -> str:
    route = state.get('route', 'retrieve')
    if route == 'tool':
        return 'tool'
    elif route == 'memory_only':
        return 'skip'
    return 'retrieve'


def eval_decision(state: CapstoneState) -> str:
    score   = state.get('faithfulness', 1.0)
    retries = state.get('eval_retries', 0)
    if score < 0.7 and retries < MAX_EVAL_RETRIES:
        return 'answer'
    return 'save'


# ─────────────────────────────────────────────────────────────────
# BUILD APP
# ─────────────────────────────────────────────────────────────────

def build_app():
    """
    Initialises all components and returns (compiled_app, embedder, collection).
    Call with @st.cache_resource to avoid re-initialisation on each Streamlit rerun.
    """
    # 1. LLM
    llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0)

    # 2. Embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. ChromaDB
    chroma_client = chromadb.Client()
    collection    = chroma_client.create_collection(name='medicare_kb')
    texts  = [d['text']              for d in KNOWLEDGE_BASE]
    ids    = [d['id']                for d in KNOWLEDGE_BASE]
    metas  = [{'topic': d['topic']}  for d in KNOWLEDGE_BASE]
    embs   = embedder.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embs, ids=ids, metadatas=metas)

    # 4. Build node functions
    _memory_node    = make_memory_node()
    _router_node    = make_router_node(llm)
    _retrieval_node = make_retrieval_node(embedder, collection)
    _answer_node    = make_answer_node(llm)
    _eval_node      = make_eval_node(llm)

    # 5. Graph
    graph = StateGraph(CapstoneState)
    graph.add_node('memory',   _memory_node)
    graph.add_node('router',   _router_node)
    graph.add_node('retrieve', _retrieval_node)
    graph.add_node('skip',     skip_retrieval_node)
    graph.add_node('tool',     tool_node)
    graph.add_node('answer',   _answer_node)
    graph.add_node('eval',     _eval_node)
    graph.add_node('save',     save_node)

    graph.set_entry_point('memory')
    graph.add_edge('memory',   'router')
    graph.add_edge('retrieve', 'answer')
    graph.add_edge('skip',     'answer')
    graph.add_edge('tool',     'answer')
    graph.add_edge('answer',   'eval')
    graph.add_edge('save',     END)

    graph.add_conditional_edges(
        'router', route_decision,
        {'retrieve': 'retrieve', 'skip': 'skip', 'tool': 'tool'}
    )
    graph.add_conditional_edges(
        'eval', eval_decision,
        {'answer': 'answer', 'save': 'save'}
    )

    app = graph.compile(checkpointer=MemorySaver())
    return app, embedder, collection


# ─────────────────────────────────────────────────────────────────
# QUICK TEST (run directly: python agent.py)
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Building app...')
    app, embedder, collection = build_app()
    print('✅ App built successfully.\n')

    def ask(question, thread_id):
        config = {'configurable': {'thread_id': thread_id}}
        state  = CapstoneState(
            question=question, messages=[], route='', retrieved='',
            sources=[], tool_result='', answer='', faithfulness=0.0,
            eval_retries=0, user_name=None
        )
        return app.invoke(state, config=config)

    tid = str(uuid.uuid4())
    questions = [
        'What are the OPD timings?',
        'Is the OPD open right now?',
        'How much is the cardiology consultation fee?',
    ]
    for q in questions:
        print(f'Q: {q}')
        r = ask(q, tid)
        print(f"Route: {r['route']} | Faith: {r['faithfulness']:.2f}")
        print(f"A: {r['answer']}\n{'-'*60}")

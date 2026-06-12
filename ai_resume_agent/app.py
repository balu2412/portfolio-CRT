import os
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables (if any)
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Resume Screening & Interview Prep Agent",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils import extract_text_from_pdf, extract_text_from_docx, get_sample_resume, get_sample_job_description
from graph import build_workflow, AgentState

# Load Custom CSS Styling
try:
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    # Fallback to default if style.css is not readable
    pass

# Session State Initialization
if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = ""
if "job_description" not in st.session_state:
    st.session_state["job_description"] = ""
if "results" not in st.session_state:
    st.session_state["results"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False

# App Header
st.markdown('<h1 class="header-gradient" style="margin-bottom: 5px;">AI Resume Screening & Interview Prep</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 25px;">A multi-agent recruitment and coaching pipeline orchestrated with LangGraph & Gemini/OpenAI</p>', unsafe_allow_html=True)

# Sidebar - Settings & Keys
with st.sidebar:
    st.markdown('<h2 style="font-size: 1.3rem; color: #a78bfa; margin-top: 0;">🛠️ Agent Configuration</h2>', unsafe_allow_html=True)
    
    # 1. API Provider
    provider_opt = st.selectbox(
        "Select LLM Provider",
        options=["Ollama (Local LLM)", "Mock Demo Mode (No API Key Required)", "Google Gemini", "OpenAI"],
        index=0
    )
    
    if provider_opt == "Google Gemini":
        provider = "gemini"
        model_name = st.selectbox(
            "Select Model",
            options=["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            index=0,
            help="gemini-2.0-flash is the recommended default. gemini-1.5-flash is extremely fast, while gemini-1.5-pro is more thorough."
        )
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    elif provider_opt == "OpenAI":
        provider = "openai"
        model_name = st.selectbox(
            "Select Model",
            options=["gpt-4o-mini", "gpt-4o"],
            index=0,
            help="gpt-4o-mini is cost-effective, while gpt-4o is state-of-the-art."
        )
        env_key = os.environ.get("OPENAI_API_KEY", "")
    elif provider_opt == "Ollama (Local LLM)":
        provider = "ollama"
        model_name = st.text_input(
            "Ollama Model Name",
            value="llama3",
            help="Enter the name of the pulled Ollama model (e.g. llama3, mistral)."
        )
        env_key = "ollama"
    else:
        provider = "mock"
        model_name = "mock"
        env_key = "mock"

    # 3. API Key Input
    if provider not in ["mock", "ollama"]:
        api_key_input = st.text_input(
            f"{provider_opt} API Key",
            value=env_key,
            type="password",
            help=f"Enter your {provider_opt} API Key. Leave empty if set in environment variables."
        )
    else:
        api_key_input = "mock"
    
    st.markdown("---")
    
    # Quick Test Option
    st.markdown('<h3 style="font-size: 1rem; color: #94a3b8; margin-bottom: 10px;">💡 Demonstration</h3>', unsafe_allow_html=True)
    load_sample_btn = st.button("Load Mock Resume & Job Desc")
    
    if load_sample_btn:
        st.session_state["resume_text"] = get_sample_resume()
        st.session_state["job_description"] = get_sample_job_description()
        st.success("Sample data loaded successfully!")
        st.rerun()
        
    st.markdown(
        '<div style="background-color: rgba(99, 102, 241, 0.1); padding: 12px; border-radius: 8px; border: 1px dashed rgba(99, 102, 241, 0.3); margin-top: 30px;">'
        '<p style="font-size: 0.8rem; color: #cbd5e1; margin: 0; line-height: 1.4;">'
        '<strong>How it works:</strong><br>'
        '1. <strong>Resume Parser</strong> extracts profile structures.<br>'
        '2. <strong>Skill Analyzer</strong> checks requirements & gaps.<br>'
        '3. <strong>Job Matcher</strong> scores overall compatibility.<br>'
        '4. <strong>Interview Coach</strong> devises custom questions.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

# Main Form inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 style="font-size: 1.25rem;">📄 Candidate Resume</h3>', unsafe_allow_html=True)
    
    # Document Uploader
    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_type = file_name.split(".")[-1].lower()
        
        with st.spinner("Extracting document text..."):
            if file_type == "pdf":
                extracted_text = extract_text_from_pdf(uploaded_file)
            elif file_type == "docx":
                extracted_text = extract_text_from_docx(uploaded_file)
            else:
                extracted_text = uploaded_file.read().decode("utf-8")
            
            if extracted_text.startswith("Error"):
                st.error(extracted_text)
            else:
                st.session_state["resume_text"] = extracted_text
                st.success(f"Successfully extracted {file_name}!")
                
    if st.session_state["resume_text"]:
        word_count = len(st.session_state["resume_text"].split())
        st.markdown(
            f"""
            <div style="background-color: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981; padding: 10px 14px; border-radius: 6px; font-size: 0.9rem; color: #a7f3d0; margin-top: 10px;">
                📄 Resume text loaded ({word_count} words)
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
    st.markdown('<h3 style="font-size: 1.25rem;">💼 Job Role & Description</h3>', unsafe_allow_html=True)
    
    job_description_area = st.text_area(
        "Paste Job Role or Description requirements here",
        value=st.session_state["job_description"],
        height=200,
        placeholder="Senior Full Stack Engineer...\nWe are looking for someone with 5+ years of Python and React experience...",
        key="job_description_input"
    )
    if job_description_area != st.session_state["job_description"]:
        st.session_state["job_description"] = job_description_area

# Run screening button
submit_btn = st.button("🚀 Screen Resume & Prepare Candidate", use_container_width=True)

if submit_btn:
    api_key = api_key_input.strip()
    if not api_key:
        st.error(f"Please enter your {provider_opt} API Key in the sidebar configuration.")
    elif not st.session_state["resume_text"].strip():
        st.error("Please upload or paste a resume first.")
    elif not st.session_state["job_description"].strip():
        st.error("Please enter a job description.")
    else:
        # We are ready to run the LangGraph!
        st.session_state["is_processing"] = True
        st.session_state["results"] = None
        
        # Build initial state
        initial_state = {
            "resume_text": st.session_state["resume_text"],
            "job_description": st.session_state["job_description"],
            "parsed_resume": {},
            "skill_analysis": {},
            "job_match": {},
            "interview_prep": {},
            "api_key": api_key,
            "api_provider": provider,
            "model_name": model_name,
            "errors": []
        }
        
        # Status update indicators
        status_box = st.empty()
        progress_bar = st.progress(0.0)
        
        try:
            workflow = build_workflow()
            
            # Step by step execution and progress updates
            current_state = initial_state
            
            # Since LangGraph execution is fast, we stream using app.stream()
            nodes_order = ["resume_parser", "skill_analyzer", "job_matcher", "interview_coach"]
            labels = {
                "resume_parser": "🤖 [1/4] Parsing candidate credentials & resume sections...",
                "skill_analyzer": "📊 [2/4] Analyzing skills & mapping gaps...",
                "job_matcher": "💼 [3/4] Evaluating job fit & calculating score...",
                "interview_coach": "🎤 [4/4] generating mock interview simulator & guide..."
            }
            
            steps_completed = 0
            
            for event in workflow.stream(initial_state):
                for node_name, state_update in event.items():
                    steps_completed += 1
                    pct = float(steps_completed) / len(nodes_order)
                    progress_bar.progress(pct)
                    
                    if node_name in labels:
                        status_box.markdown(f'<div class="custom-alert alert-info">⌛ <strong>In Progress:</strong> {labels[node_name]}</div>', unsafe_allow_html=True)
                    
                    # Merge update into current state
                    if isinstance(state_update, dict):
                        for key, val in state_update.items():
                            current_state[key] = val
            
            if current_state.get("errors"):
                # Display errors nicely
                st.session_state["is_processing"] = False
                for err in current_state["errors"]:
                    st.error(err)
            else:
                status_box.markdown('<div class="custom-alert alert-success">✅ <strong>Success:</strong> Multi-Agent screening completed!</div>', unsafe_allow_html=True)
                progress_bar.progress(1.0)
                st.session_state["results"] = current_state
                st.session_state["is_processing"] = False
                st.rerun()
                
        except Exception as e:
            st.session_state["is_processing"] = False
            import traceback
            tb_str = traceback.format_exc()
            print("Full Traceback:")
            print(tb_str)
            with open("error.log", "w") as f:
                f.write(tb_str)
            st.error(f"Error compiling or executing workflow: {str(e)}")

# Display Results
results = st.session_state["results"]
if results and not st.session_state["is_processing"]:
    st.markdown("---")
    st.markdown('<h2 style="color: #a78bfa; margin-bottom: 20px;">📊 AI Evaluation Reports</h2>', unsafe_allow_html=True)
    
    # Extract results
    parsed = results.get("parsed_resume", {})
    skills_rep = results.get("skill_analysis", {})
    job_match = results.get("job_match", {})
    prep = results.get("interview_prep", {})
    
    # 5 Tab Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Executive Dashboard",
        "📄 Parsed Profile",
        "📊 Skill Gaps",
        "💼 Job Fit Match",
        "🎤 Prep Coach"
    ])
    
    # --- TAB 1: EXECUTIVE DASHBOARD ---
    with tab1:
        col_dash_left, col_dash_right = st.columns([1, 2])
        
        with col_dash_left:
            # Score Gauge
            score = job_match.get("match_percentage", 0)
            deg = score * 3.6
            st.markdown(
                f"""
                <div class="glass-card" style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <div class="score-container">
                        <div class="circle-outer" style="--score-deg: {deg}deg;">
                            <div class="circle-inner">
                                <div class="score-text">{score}%</div>
                                <div class="score-label">Match Score</div>
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; text-align: center;">
                        <span style="font-size: 1.2rem; font-weight: 600; color: #a78bfa;">{job_match.get('verdict', 'N/A')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Candidate Basic Info Box
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="margin-top: 0; color: #a78bfa;">👤 Candidate Profile</h4>
                    <p style="margin-bottom: 8px;"><strong>Name:</strong> {parsed.get('candidate_name', 'N/A')}</p>
                    <p style="margin-bottom: 8px;"><strong>Email:</strong> {parsed.get('email', 'N/A')}</p>
                    <p style="margin-bottom: 0;"><strong>Phone:</strong> {parsed.get('phone', 'N/A')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_dash_right:
            # Summary
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="margin-top: 0; color: #a78bfa;">📝 Candidate Summary</h4>
                    <p style="line-height: 1.6; color: #cbd5e1; font-style: italic;">"{parsed.get('summary', 'No summary available.')}"</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Strengths and Weaknesses High level overview
            col_sw_l, col_sw_r = st.columns(2)
            
            with col_sw_l:
                st.markdown('<h5 style="color: #34d399; margin-bottom: 10px;">🌟 Key Strengths</h5>', unsafe_allow_html=True)
                for strg in job_match.get("strengths", [])[:3]:
                    st.markdown(f"- {strg}")
            
            with col_sw_r:
                st.markdown('<h5 style="color: #fb7185; margin-bottom: 10px;">⚠️ Gaps & Concerns</h5>', unsafe_allow_html=True)
                for weak in job_match.get("weaknesses", [])[:3]:
                    st.markdown(f"- {weak}")
                    
    # --- TAB 2: PARSED PROFILE ---
    with tab2:
        col_prof_l, col_prof_r = st.columns([1, 2])
        
        with col_prof_l:
            st.markdown('<h4 style="color: #a78bfa; margin-top: 0;">🛠️ Technical & Soft Skills</h4>', unsafe_allow_html=True)
            skills = parsed.get("skills", [])
            if skills:
                badges_html = "".join([f'<span class="badge badge-general">{skill}</span>' for skill in skills])
                st.markdown(f'<div class="badge-container">{badges_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No skills extracted.")
                
            # Education
            st.markdown('<h4 style="color: #a78bfa; margin-top: 25px;">🎓 Education</h4>', unsafe_allow_html=True)
            edu_items = parsed.get("education", [])
            if edu_items:
                for edu in edu_items:
                    st.markdown(
                        f"""
                        <div class="timeline-item">
                            <div class="timeline-dot"></div>
                            <div class="timeline-header">
                                <span class="timeline-title">{edu.get('degree', 'N/A')}</span>
                                <span class="timeline-date">{edu.get('year', 'N/A')}</span>
                            </div>
                            <div class="timeline-company">{edu.get('institution', 'N/A')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No education information found.")
                
        with col_prof_r:
            # Experience Timeline
            st.markdown('<h4 style="color: #a78bfa; margin-top: 0;">💼 Work Experience</h4>', unsafe_allow_html=True)
            exp_items = parsed.get("experience", [])
            if exp_items:
                for exp in exp_items:
                    desc_bullets = "".join([f"<li>{item}</li>" for item in exp.get("description", [])])
                    st.markdown(
                        f"""
                        <div class="timeline-item">
                            <div class="timeline-dot"></div>
                            <div class="timeline-header">
                                <span class="timeline-title">{exp.get('role', 'N/A')}</span>
                                <span class="timeline-date">{exp.get('duration', 'N/A')}</span>
                            </div>
                            <div class="timeline-company">{exp.get('company', 'N/A')}</div>
                            <ul class="timeline-desc" style="padding-left: 18px; margin-top: 8px;">
                                {desc_bullets}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No experience information found.")
                
            # Projects
            st.markdown('<h4 style="color: #a78bfa; margin-top: 25px;">🚀 Projects</h4>', unsafe_allow_html=True)
            proj_items = parsed.get("projects", [])
            if proj_items:
                for proj in proj_items:
                    techs = "".join([f'<span class="badge badge-general" style="font-size: 0.75rem; padding: 3px 8px;">{t}</span>' for t in proj.get("technologies", [])])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="padding: 16px; margin-bottom: 15px;">
                            <h5 style="margin-top: 0; margin-bottom: 8px; font-weight: 600;">{proj.get('name', 'N/A')}</h5>
                            <div class="badge-container" style="margin: 5px 0 10px 0;">{techs}</div>
                            <p style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0;">{proj.get('description', '')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No projects found.")
                
    # --- TAB 3: SKILL GAPS ---
    with tab3:
        st.markdown('<h4 style="color: #a78bfa; margin-top: 0;">📊 Requirements Skill Match Analysis</h4>', unsafe_allow_html=True)
        
        col_gap_l, col_gap_r = st.columns(2)
        
        with col_gap_l:
            st.markdown('##### 🟢 Matched Skills')
            match_skills = skills_rep.get("matching_skills", [])
            if match_skills:
                badges_m = "".join([f'<span class="badge badge-match">{sk}</span>' for sk in match_skills])
                st.markdown(f'<div class="badge-container">{badges_m}</div>', unsafe_allow_html=True)
            else:
                st.info("No matching skills identified.")
                
        with col_gap_r:
            st.markdown('##### 🔴 Missing Skills / Requirements')
            miss_skills = skills_rep.get("missing_skills", [])
            if miss_skills:
                badges_ms = "".join([f'<span class="badge badge-missing">{sk}</span>' for sk in miss_skills])
                st.markdown(f'<div class="badge-container">{badges_ms}</div>', unsafe_allow_html=True)
            else:
                st.success("No missing skills! Candidate matches all major qualifications.")
                
        st.markdown("---")
        st.markdown('<h4 style="color: #a78bfa;">💡 Recommendations to Bridge the Gaps</h4>', unsafe_allow_html=True)
        suggestions = skills_rep.get("improvement_suggestions", [])
        if suggestions:
            for i, sug in enumerate(suggestions):
                st.markdown(
                    f"""
                    <div style="background-color: rgba(30, 41, 59, 0.5); padding: 12px 18px; border-radius: 8px; border-left: 3px solid #6366f1; margin-bottom: 10px;">
                        <span style="font-weight: 500; color: #f8fafc;">{sug}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No suggestions necessary.")
            
    # --- TAB 4: JOB FIT MATCH ---
    with tab4:
        st.markdown('<h4 style="color: #a78bfa; margin-top: 0;">💼 Job Compatibility Assessment</h4>', unsafe_allow_html=True)
        
        col_match_l, col_match_r = st.columns([1, 1])
        
        with col_match_l:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h5 style="color: #a78bfa; margin-top: 0;">Evaluation Summary</h5>
                    <p style="font-size: 1.1rem; margin-bottom: 10px;"><strong>Overall Score:</strong> <span style="font-size: 1.4rem; color: #818cf8; font-weight: 700;">{job_match.get('match_percentage', 0)}%</span></p>
                    <p style="font-size: 1.1rem; margin-bottom: 10px;"><strong>Verdict:</strong> <span style="font-weight: 600;">{job_match.get('verdict', 'N/A')}</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Other roles
            st.markdown('<h5 style="color: #a78bfa;">💼 Suggested Alternative Roles</h5>', unsafe_allow_html=True)
            roles = job_match.get("suitable_roles", [])
            if roles:
                for r in roles:
                    st.markdown(f"- **{r}**")
            else:
                st.info("No alternative roles suggested.")
                
        with col_match_r:
            st.markdown('<h5 style="color: #34d399; margin-top: 0;">🌟 Fit Strengths</h5>', unsafe_allow_html=True)
            for s in job_match.get("strengths", []):
                st.markdown(f"✅ {s}")
                
            st.markdown('<h5 style="color: #fb7185; margin-top: 20px;">⚠️ Risk Areas / Gaps</h5>', unsafe_allow_html=True)
            for w in job_match.get("weaknesses", []):
                st.markdown(f"❌ {w}")
                
    # --- TAB 5: PREP COACH ---
    with tab5:
        st.markdown('<h4 style="color: #a78bfa; margin-top: 0;">🎤 Personalized Mock Interview Prep Coach</h4>', unsafe_allow_html=True)
        st.markdown("Below is a set of simulated interview questions, structured strategies, and custom responses tailored specifically to your background and the target role.")
        
        # Technical Questions
        st.markdown('<h5 style="color: #818cf8; font-size: 1.15rem; margin-top: 20px; border-bottom: 1px solid #334155; padding-bottom: 5px;">💻 Technical & Role-Specific Questions</h5>', unsafe_allow_html=True)
        tech_qs = prep.get("technical_questions", [])
        if tech_qs:
            for idx, q_item in enumerate(tech_qs):
                with st.expander(f"Q{idx+1}: {q_item.get('question')} ({q_item.get('difficulty', 'Medium')})"):
                    st.markdown(f"**Type:** {q_item.get('type')}")
                    st.markdown(f"**Best Approach:** *{q_item.get('best_approach')}*")
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.2); padding: 16px; border-radius: 8px; margin-top: 10px;">
                            <strong style="color: #818cf8;">Sample Answer:</strong><br>
                            <p style="margin-top: 5px; color: #e2e8f0; line-height: 1.5; font-size: 0.95rem;">{q_item.get('sample_answer')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("No technical questions generated.")
            
        # Behavioral Questions
        st.markdown('<h5 style="color: #818cf8; font-size: 1.15rem; margin-top: 30px; border-bottom: 1px solid #334155; padding-bottom: 5px;">🤝 Behavioral & HR Questions</h5>', unsafe_allow_html=True)
        beh_qs = prep.get("behavioral_questions", [])
        if beh_qs:
            for idx, q_item in enumerate(beh_qs):
                with st.expander(f"Q{idx+1}: {q_item.get('question')} ({q_item.get('difficulty', 'Medium')})"):
                    st.markdown(f"**Type:** {q_item.get('type')}")
                    st.markdown(f"**Best Approach:** *{q_item.get('best_approach')}*")
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.2); padding: 16px; border-radius: 8px; margin-top: 10px;">
                            <strong style="color: #818cf8;">Sample Answer:</strong><br>
                            <p style="margin-top: 5px; color: #e2e8f0; line-height: 1.5; font-size: 0.95rem;">{q_item.get('sample_answer')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("No behavioral questions generated.")
            
        # General Tips
        st.markdown('<h5 style="color: #a78bfa; font-size: 1.15rem; margin-top: 30px;">💡 General Interview Strategies</h5>', unsafe_allow_html=True)
        general_tips = prep.get("general_tips", [])
        if general_tips:
            for tip in general_tips:
                st.markdown(f"📍 {tip}")
        else:
            st.info("No general tips available.")
else:
    if not st.session_state["is_processing"]:
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; padding: 40px; background-color: rgba(30, 41, 59, 0.4); border-radius: 12px; border: 1px dashed #334155;">
                <h4 style="color: #cbd5e1; margin-top: 0;">No Analysis Performed Yet</h4>
                <p style="color: #94a3b8;">Provide your API keys, upload/paste a candidate resume and job description, then click <strong>Screen Resume</strong> to generate the evaluation reports.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

import socket

def is_cloud():
    hostname = socket.gethostname()
    return "streamlit" in hostname.lower() or "railway" in hostname.lower() or "render" in hostname.lower()

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import pytesseract
from pdf2image import convert_from_bytes
import io

# NLTK setup

nltk_pkgs = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger"]

for pkg in nltk_pkgs:
    try:
        if pkg in ("punkt", "punkt_tab"):
            nltk.data.find(f"tokenizers/{pkg}")
        elif pkg == "averaged_perceptron_tagger":
            nltk.data.find("taggers/averaged_perceptron_tagger")
        else:  # stopwords, wordnet, omw-1.4
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:                        
            pass

# Page config & Sidebar UI
st.set_page_config(
    page_title="Resume Job Match — OCR + Synonyms + Suggestions",
    page_icon="📄",
    layout="wide"
)

with st.sidebar:
    st.header("About")
    st.info("""
    This tool extracts text from resumes (PDF or images), falls back to OCR if needed,
    expands job keywords using synonyms, and generates short one-line resume suggestions.
    """)

    # How it works in its own expander
    with st.expander("How it works", expanded=True):
        st.write("""
        1. Upload your resume (PDF or image)  
        2. Paste the job description  
        3. Click **Analyze Match**  
        4. Review score, missing keywords and suggestions
        """)

    # Settings
    st.markdown("### Settings")
    if is_cloud():
        ocr_enabled = False
        st.info("OCR disabled on cloud. Upload text PDFs only.")
    else:
        ocr_enabled = st.checkbox("Enable OCR fallback (requires tesseract + poppler)", value=True)

    synonym_enabled = st.checkbox("Enable synonym expansion (WordNet + curated tech map)", value=True)
    top_k = st.number_input("Top keywords to extract (job)", min_value=3, max_value=50, value=12, step=1)
    highlight_toggle = st.checkbox("Highlight matched keywords in preview", value=True)
    sentences_to_show = st.slider("Top matching sentences to show", min_value=1, max_value=6, value=3)
    st.markdown("---")
    st.caption("Notes: OCR is slower. Synonyms improve recall but may introduce false positives.")

def extract_text_from_pdf(uploaded_file, ocr_if_empty=True):
    """
    Extract text using PyPDF2; fallback to OCR (pdf2image + pytesseract)
    if empty and allowed.
    """
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = None
            if page_text:
                text += page_text + " "
        text = text.strip()
    except Exception:
        text = ""

    # OCR fallback for scanned PDFs
    if (not text or len(text) < 50) and ocr_if_empty and ocr_enabled:
        try:
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            images = convert_from_bytes(pdf_bytes, dpi=300)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + " "
            return ocr_text.strip()
        except Exception:
            # if OCR also fails, just return whatever we had
            return text

    return text


def extract_text_from_image(uploaded_file):
    """Extract text from an uploaded image (jpg/png) using OCR."""
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        file_bytes = uploaded_file.read()        
        images = convert_from_bytes(file_bytes)
        if not images:
            return ""
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + " "
        return text.strip()
    except Exception:        
        try:
            from PIL import Image
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(file_bytes))
            return pytesseract.image_to_string(img).strip()
        except Exception:
            return ""


def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    if not text:
        return ""
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(filtered)


def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    if not resume_processed or not job_processed:
        return 0.0, resume_processed, job_processed
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed

# Tech synonyms map + WordNet helper
TECH_SYNONYMS = {
    "aws": ["amazon web services"],
    "amazon web services": ["aws"],
    "tensorflow": ["tf", "tensorflow"],
    "pytorch": ["torch", "pytorch"],
    "nlp": ["natural language processing"],
    "natural language processing": ["nlp"],
    "docker": ["containers", "containerization"],
    "kubernetes": ["k8s"],
    "sql": ["structured query language"],
    "restful api": ["api", "rest api", "restful apis"],
    "api": ["rest api", "restful api"],
    "git": ["version control", "git"],
    "ci/cd": ["continuous integration", "continuous deployment", "ci cd"],
    "spark": ["apache spark"],
    "hadoop": ["apache hadoop"]
}

def get_wordnet_synonyms(word):
    syns = set()
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name != word.lower():
                    syns.add(name)
    except Exception:
        pass
    return list(syns)


def expand_keyword_variants(keyword):
    kw = keyword.lower().strip()
    variants = set([kw])

    if kw in TECH_SYNONYMS:
        for v in TECH_SYNONYMS[kw]:
            variants.add(v.lower())
    for k, vals in TECH_SYNONYMS.items():
        if kw in vals:
            variants.add(k.lower())
            for v in vals:
                variants.add(v.lower())

    # WordNet for single tokens
    tokens = kw.split()
    if synonym_enabled and len(tokens) == 1 and tokens[0].isalpha() and len(tokens[0]) > 2:
        for s in get_wordnet_synonyms(tokens[0]):
            variants.add(s)

    return sorted(list(variants), key=lambda x: -len(x))


def token_set(text):
    return set(word_tokenize(clean_text(text)))


def find_missing_keywords_expanded(keywords, resume_text):
    res_tokens = token_set(resume_text)
    present = []
    missing = []
    detail = {}

    for kw in keywords:
        variants = expand_keyword_variants(kw)
        matched_variant = None
        for v in variants:
            v_tokens = set(word_tokenize(v))
            if v_tokens and v_tokens.issubset(res_tokens):
                matched_variant = v
                break
        if matched_variant:
            present.append(kw)
            detail[kw] = ("present", matched_variant)
        else:
            missing.append(kw)
            detail[kw] = ("missing", variants[:4])

    return present, missing, detail


def extract_top_keywords(job_text, k=10):
    job_clean = clean_text(job_text)
    if not job_clean:
        return []
    vec = TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1, 2))
    tfidf = vec.fit_transform([job_clean])
    feature_array = vec.get_feature_names_out()
    tfidf_sorting = tfidf.toarray().flatten().argsort()[::-1]
    top_n = [feature_array[i] for i in tfidf_sorting][:k]
    return top_n


def extract_common_verbs(resume_text, top_n=5):
    # Resilient verb extractor
    import re as _re
    tokens = []
    try:
        tokens = nltk.word_tokenize(resume_text.lower())
    except Exception:
        tokens = _re.findall(r'\w+', resume_text.lower())

    try:
        tagged = nltk.pos_tag(tokens)
        verbs = [w for w, pos in tagged if pos.startswith('VB')]
        c = Counter(verbs)
        common = [v for v, _ in c.most_common(top_n)]
    except Exception:
        fallback_verbs = [
            "developed", "designed", "implemented", "built", "deployed",
            "created", "engineered", "led", "improved", "optimized",
            "managed", "tested", "maintained", "analyzed", "researched", "worked"
        ]
        found = [v for v in fallback_verbs if _re.search(r'\b' + _re.escape(v) + r'\b', resume_text.lower())]
        common = found[:top_n] or ["worked"]

    priority = [v for v in [
        "developed", "designed", "implemented", "built", "deployed", "created", "engineered"
    ] if v in common]
    return priority[:top_n] if priority else common[:top_n]


def extract_project_names(resume_text):
    lines = resume_text.splitlines()
    projects = []
    for i, line in enumerate(lines):
        if re.search(r'project', line, re.I):
            for j in range(i + 1, min(i + 6, len(lines))):
                s = lines[j].strip()
                if s.startswith('-') or s.startswith('•'):
                    projects.append(re.sub(r'^[\-•\s]+', '', s)[:80])
        m = re.search(r'project[:\-]\s*(.+)', line, re.I)
        if m:
            projects.append(m.group(1).strip()[:80])
    return projects[:3]


def generate_suggestion_for_keyword(keyword, resume_text, verbs_list=None, projects=None):
    verbs_list = verbs_list or extract_common_verbs(resume_text)
    projects = projects or extract_project_names(resume_text)
    verb = verbs_list[0] if verbs_list else "Worked"
    if projects:
        project = projects[0]
        suggestion = f'Add: "{verb.capitalize()} {keyword} in the {project} project (e.g., used {keyword} to ... )."'
    else:
        suggestion = f'Add: "{verb.capitalize()} experience with {keyword} (e.g., implemented or integrated {keyword} in a project)."'
    return suggestion


def top_matching_sentences(resume_text, job_text, top_n=3):
    sents = sent_tokenize(resume_text)
    if not sents:
        return []
    vec = TfidfVectorizer(stop_words='english')
    docs = sents + [job_text]
    tfidf = vec.fit_transform(docs)
    job_vec = tfidf[-1]
    sent_vecs = tfidf[:-1]
    sims = cosine_similarity(sent_vecs, job_vec).flatten()
    best_idx = sims.argsort()[::-1][:top_n]
    return [(sents[i], round(float(sims[i]) * 100, 2)) for i in best_idx]


def highlight_text(text, keywords):
    if not text:
        return ""
    t = text
    for kw in sorted(keywords, key=len, reverse=True):
        pattern = r'(?i)\b' + re.escape(kw) + r'\b'
        t = re.sub(pattern, f"<mark>{kw}</mark>", t)
    return t

# Main UI & logic
st.title("Resume Job Match — OCR + Synonyms + Suggestions")
st.write("Upload resume (PDF or image), paste job description, then click Analyze. Use sidebar to tune options.")

uploaded_file = st.file_uploader("Upload your resume (PDF or Image)", type=['pdf', 'jpg', 'jpeg', 'png'])
job_description = st.text_area("Paste the job description", height=220)

if st.button("Analyze Match"):

    # Basic validations
    if not uploaded_file:
        st.warning("Please upload your resume")
        st.stop()

    if not job_description or job_description.strip() == "":
        st.warning("Please paste the job description")
        st.stop()

    with st.spinner("Analyzing... (this may take longer if OCR is used)"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        # Detect file type
        file_type = uploaded_file.name.split('.')[-1].lower()
        resume_text = ""

        # Handle images (JPG/PNG)
        if file_type in ["jpg", "jpeg", "png"]:
            if is_cloud():
                st.warning(
                    "⚠️ JPG / PNG resume upload is not supported on cloud version.\n\n"
                    "Please use a text-based PDF OR run this app locally to enable OCR for images."
                )
                st.stop()
            else:
                # Local machine: run OCR on image
                resume_text = extract_text_from_image(uploaded_file)

        # Handle PDFs
        elif file_type == "pdf":
            resume_text = extract_text_from_pdf(uploaded_file, ocr_if_empty=True)

        else:
            st.error("Unsupported file type. Upload a PDF or an image resume (JPG/PNG).")
            st.stop()

    # After spinner: verify extracted text
    if not resume_text or len(resume_text.strip()) == 0:
        st.error(
            "Could not extract text from the uploaded file.\n"
            "- If it's a scanned PDF/image, ensure Tesseract & Poppler are installed.\n"
            "- Also enable OCR in the sidebar (if you are running locally)."
        )
        st.stop()

    # Compute similarity and analytics
    similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)
    top_keywords = extract_top_keywords(job_description, k=top_k)

    if synonym_enabled:
        present_kws, missing_kws, detail_map = find_missing_keywords_expanded(top_keywords, resume_text)
    else:
        present_kws = []
        missing_kws = []
        detail_map = {}
        res_tokens = token_set(resume_text)
        for kw in top_keywords:
            kw_tokens = set(word_tokenize(kw))
            if kw_tokens and kw_tokens.issubset(res_tokens):
                present_kws.append(kw)
                detail_map[kw] = ("present", kw)
            else:
                missing_kws.append(kw)
                detail_map[kw] = ("missing", [kw])

    top_sents = top_matching_sentences(resume_text, job_description, top_n=sentences_to_show)
    verbs = extract_common_verbs(resume_text)
    projects = extract_project_names(resume_text)
    suggestions = [
        generate_suggestion_for_keyword(kw, resume_text, verbs_list=verbs, projects=projects)
        for kw in missing_kws
    ]

    # Display results
    st.subheader("Results")
    st.metric("Match Score", f"{similarity_score:.2f}%")

    # Bar visualization
    fig, ax = plt.subplots(figsize=(11, 0.2))
    colors = ['#ff4b4b', '#ffa726', '#0f9d58']
    color_index = min(int(similarity_score // 33), 2)
    ax.barh([0], [similarity_score], color=colors[color_index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Match percentage")
    ax.set_yticks([])
    ax.set_title("Resume Job Match")
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig, use_container_width=True)

    if similarity_score < 40:
        st.warning("Low Match — consider tailoring your resume more closely to this job.")
    elif similarity_score < 70:
        st.info("Good Match — your resume aligns fairly well with the job.")
    else:
        st.success("Excellent Match! Your resume strongly aligns with the job description.")

    # Keywords
    st.subheader("Top job keywords (extracted)")
    st.write(", ".join(top_keywords) if top_keywords else "No keywords found.")
    st.write(f"Keywords present in resume: {len(present_kws)} / {len(top_keywords)}")

    if missing_kws:
        st.error("Missing keywords (consider adding these or close synonyms):")
        for kw in missing_kws:
            tag = detail_map.get(kw, ("missing", []))
            if isinstance(tag[1], list):
                variants = ", ".join(tag[1])
            else:
                variants = tag[1]
            st.write(f"- **{kw}** — variants: {variants}")
    else:
        st.success("No missing top keywords detected!")

    # Suggestions
    st.subheader("One-line resume suggestions (template-based)")
    if suggestions:
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.write("No suggestions — your resume already contains the top keywords.")

    # Top matching sentences
    st.subheader("Top matching resume sentences")
    if top_sents:
        for i, (sent, score) in enumerate(top_sents, start=1):
            st.markdown(f"**{i}.** ({score:.2f}%) {sent}")
    else:
        st.write("No matching sentences found or resume is too short.")

    # Highlighted previews
    if highlight_toggle:
        st.subheader("Highlighted preview (job description)")
        highlighted_job = highlight_text(job_description, present_kws)
        st.markdown(highlighted_job, unsafe_allow_html=True)

        st.subheader("Highlighted preview (resume) — preview truncated to first 4000 chars")
        preview_resume = resume_text[:4000]
        highlighted_resume = highlight_text(preview_resume, present_kws)
        st.markdown(highlighted_resume, unsafe_allow_html=True)
        if len(resume_text) > len(preview_resume):
            st.write("... (preview truncated)")

    # Downloadable report
    report_lines = [
        f"Match Score: {similarity_score:.2f}%",
        "",
        "Top keywords:",
        ", ".join(top_keywords),
        "",
        "Keywords present:",
        ", ".join(present_kws) if present_kws else "None",
        "",
        "Missing keywords:",
        ", ".join(missing_kws) if missing_kws else "None",
        "",
        "Suggestions:",
    ]
    report_lines.extend(suggestions if suggestions else ["None"])
    report_text = "\n".join(report_lines)

    st.download_button(
        "Download report (txt)",
        report_text,
        file_name="resume_match_report.txt",
        mime="text/plain"
    )

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import subprocess
import sys
import importlib
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

try:
    import docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    spacy = None
    SPACY_AVAILABLE = False

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CONCLUDING_PATTERNS = [
    r"^pour finir\b",
    r"^pour terminer\b",
    r"^pour conclure\b",
    r"^en conclusion\b",
    r"^enfin\b",
    r"^pour terminer cette revue\b",
    r"^pour clore\b",
]
CONCLUDING_REGEXES = [re.compile(p, flags=re.I | re.U) for p in CONCLUDING_PATTERNS]

def word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())

def is_concluding_phrase(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    for rx in CONCLUDING_REGEXES:
        if rx.search(t):
            return True
    return False

def simple_tokenize_lemmatize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    cleaned = re.sub(r"[^\w'\s\-]", " ", text.lower(), flags=re.U)
    tokens = [t.strip("'\"") for t in cleaned.split() if t.strip("'\"")]
    lemmas = []
    for t in tokens:
        for suf in ("ment", "tion", "ions", "er", "ez", "ais", "ait", "ons", "ent", "es", "é", "ée", "és", "ées"):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                t = t[:-len(suf)]
                break
        lemmas.append(t)
    return lemmas

def load_spacy_fr_model():
    if not SPACY_AVAILABLE:
        return None
    candidates = ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"]
    for name in candidates:
        try:
            return spacy.load(name)
        except Exception:
            continue
    return None

def ensure_spacy_model(model_name: str = "fr_core_news_sm") -> bool:
    try:
        importlib.import_module(model_name)
        return True
    except Exception:
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
            importlib.invalidate_caches()
            importlib.import_module(model_name)
            return True
        except Exception as e:
            st.warning(f"Could not download spaCy model {model_name}: {e}")
            return False

def lemmatize_with_spacy(nlp, text: str) -> List[str]:
    if not nlp:
        return simple_tokenize_lemmatize(text)
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = (token.lemma_ or token.text).lower().strip()
        if lemma:
            lemmas.append(lemma)
    return lemmas

def compute_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return float("nan")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def read_text_from_uploaded_file(uploaded) -> str:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return uploaded.getvalue().decode("utf-8", errors="replace")
    if name.endswith(".docx") and DOCX_AVAILABLE:
        doc = docx.Document(io.BytesIO(uploaded.getvalue()))
        full = "\n\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
        return full
    try:
        return uploaded.getvalue().decode("utf-8", errors="replace")
    except Exception:
        return uploaded.getvalue().decode("latin-1", errors="replace")

def extract_paragraphs_and_transitions_from_text(text: str) -> Tuple[List[str], List[str], Dict[str, str]]:
    metadata = {}
    txt = text.replace("\r\n", "\n").replace("\r", "\n")
    lower = txt.lower()
    title_match = re.search(r"^titre\s*:\s*(.+)$", txt, flags=re.I | re.M)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    chapeau_match = re.search(r"^chapeau\s*:\s*(.+)$", txt, flags=re.I | re.M)
    if chapeau_match:
        metadata["chapeau"] = chapeau_match.group(1).strip()
    trans_pos = None
    for token in ("transitions génér", "transitions g", "transitions", "transitions generated"):
        i = lower.rfind(token)
        if i != -1:
            trans_pos = i
            break
    article_pos = lower.find("article:")
    if article_pos != -1:
        article_start = article_pos + len("article:")
    else:
        article_start = 0
    if trans_pos is None:
        article_body = txt[article_start:].strip()
        trans_block = ""
    else:
        article_body = txt[article_start:trans_pos].strip()
        trans_block = txt[trans_pos:].strip()
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', article_body) if p.strip()]
    transitions = []
    if trans_block:
        lines = trans_block.splitlines()
        start_idx = 0
        for i, ln in enumerate(lines):
            if re.search(r"transitions", ln, flags=re.I):
                start_idx = i + 1
                break
        for ln in lines[start_idx:]:
            ln = ln.strip()
            if not ln:
                continue
            ln_clean = re.sub(r'^\s*\d+[\.\)]\s*', '', ln)
            ln_clean = ln_clean.strip(" -\t")
            if ln_clean:
                transitions.append(ln_clean)
    else:
        for p in paragraphs:
            if word_count(p) <= 6 and len(p) < 120:
                transitions.append(p.strip())
    return paragraphs, transitions, metadata

def map_transitions_to_context(paragraphs: List[str], transitions: List[str]) -> List[Dict[str, Any]]:
    results = []
    for t in transitions:
        found_idx = None
        t_norm = t.strip()
        for i, p in enumerate(paragraphs):
            if p.strip() == t_norm:
                found_idx = i
                break
        if found_idx is None:
            for i, p in enumerate(paragraphs):
                if t_norm in p:
                    found_idx = i
                    break
        if found_idx is None:
            t_first = " ".join(t_norm.split()[:6]).strip()
            for i, p in enumerate(paragraphs):
                if t_first and t_first in p:
                    found_idx = i
                    break
        if found_idx is not None:
            prev_p = paragraphs[found_idx - 1] if found_idx - 1 >= 0 else ""
            next_p = paragraphs[found_idx + 1] if found_idx + 1 < len(paragraphs) else ""
            para_idx = found_idx
        else:
            prev_p = ""
            next_p = ""
            para_idx = None
        results.append({
            "transition_text": t_norm,
            "para_idx": para_idx,
            "prev_paragraph": prev_p,
            "next_paragraph": next_p
        })
    return results

@dataclass
class TransitionResult:
    article_id: Any
    para_idx: Any
    transition_text: str
    word_count: int
    is_concluding: bool
    placement_check: Optional[bool]
    repeated_lemmas: List[str]
    repetition_flag: bool
    similarity_next: Optional[float]
    similarity_prev: Optional[float]
    thematic_ok: Optional[bool]
    final_verdict: str
    failure_reasons: List[str]
    triggered_rules: List[str]

def get_fr_stopwords(nlp=None) -> set:
    base = set()
    if nlp and hasattr(nlp, "Defaults") and getattr(nlp.Defaults, "stop_words", None):
        base |= set(nlp.Defaults.stop_words)
    base |= {
        "à", "au", "aux", "avec", "ce", "cet", "cette", "ces", "cela", "ça", "ceci", "c’", "d’", "de", "des", "du",
        "dans", "en", "et", "la", "le", "les", "l’", "un", "une", "on", "ou", "où", "pour", "par", "pas", "plus",
        "que", "qui", "quoi", "qu’", "se", "sa", "son", "ses", "sur", "sans", "entre", "comme", "mais", "donc",
        "or", "ni", "car", "y", "aujourd’hui", "toute", "tous", "toutes", "tout"
    }
    return base

def content_lemmas(text: str, nlp=None, stopwords: set = None) -> list:
    stopwords = stopwords or set()
    if not text:
        return []
    if nlp:
        doc = nlp(text)
        keep_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
        out = []
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            lem = (t.lemma_ or t.text).lower().strip("’'`-–—.!,?;:()[]{}")
            if not lem or len(lem) <= 2:
                continue
            if lem in stopwords:
                continue
            if t.pos_ and t.pos_ not in keep_pos:
                continue
            out.append(lem)
        return out
    cleaned = re.sub(r"[^\w'\s-]", " ", (text or "").lower())
    toks = [tok.strip("’'`-") for tok in cleaned.split()]
    out = []
    for tok in toks:
        if len(tok) <= 2:
            continue
        if tok in stopwords:
            continue
        out.append(tok)
    return out

def analyze_group_rows(rows: List[Dict[str, Any]], nlp, embed_model,
                       thematic_delta: float = 0.0,
                       require_para_idx: bool = False) -> List[TransitionResult]:
    fr_stop = get_fr_stopwords(nlp)
    all_texts_for_lemmas = []
    for r in rows:
        t = r.get("transition_text", "") or ""
        p = r.get("prev_paragraph", "") or ""
        n = r.get("next_paragraph", "") or ""
        if t:
            all_texts_for_lemmas.append(t)
        if p:
            all_texts_for_lemmas.append(p)
        if n:
            all_texts_for_lemmas.append(n)
    lemmas_per_text = [content_lemmas(t, nlp, fr_stop) for t in all_texts_for_lemmas]
    lemma_counts = {}
    for lem_list in lemmas_per_text:
        for l in set(lem_list):
            lemma_counts[l] = lemma_counts.get(l, 0) + 1
    trans_texts = [r.get("transition_text", "") or "" for r in rows]
    trans_lemmas_list = [content_lemmas(t, nlp, fr_stop) for t in trans_texts]
    unique_texts = list({r["transition_text"] for r in rows} |
                        {r.get("prev_paragraph", "") for r in rows} |
                        {r.get("next_paragraph", "") for r in rows})
    embeddings = {}
    if embed_model is not None:
        try:
            embs = embed_model.encode(unique_texts, convert_to_numpy=True, show_progress_bar=False)
            for txt, emb in zip(unique_texts, embs):
                embeddings[txt] = emb
        except Exception as e:
            st.warning(f"Embedding model error (embeddings disabled for this article): {e}")
            embeddings = {}
    para_indices = [r.get("para_idx") for r in rows if r.get("para_idx") is not None]
    max_para_idx = max(para_indices) if para_indices else None
    results = []
    for idx, r in enumerate(rows):
        ttext = r.get("transition_text", "") or ""
        prev = r.get("prev_paragraph", "") or ""
        nxt = r.get("next_paragraph", "") or ""
        wc = word_count(ttext)
        is_conc = is_concluding_phrase(ttext)
        if max_para_idx is not None:
            placement_ok = (r.get("para_idx") is not None and r.get("para_idx") == max_para_idx)
            if require_para_idx and r.get("para_idx") is None:
                placement_ok = False
        else:
            placement_ok = (not nxt.strip())
            if require_para_idx:
                placement_ok = False
        this_lemmas = set(trans_lemmas_list[idx])
        repeated = sorted([l for l in this_lemmas if lemma_counts.get(l, 0) > 1])
        repetition_flag = len(repeated) > 0
        sim_next = sim_prev = thematic_ok = None
        if embeddings:
            emb_t = embeddings.get(ttext)
            emb_n = embeddings.get(nxt)
            emb_p = embeddings.get(prev)
            if emb_t is not None and emb_n is not None:
                sim_next = float(cosine_similarity(emb_t.reshape(1, -1), emb_n.reshape(1, -1))[0][0])
            if emb_t is not None and emb_p is not None:
                sim_prev = float(cosine_similarity(emb_t.reshape(1, -1), emb_p.reshape(1, -1))[0][0])
            if sim_next is not None and sim_prev is not None:
                thematic_ok = (sim_next - sim_prev) > thematic_delta
        failure_reasons: List[str] = []
        triggered_rules: List[str] = []
        if wc > 5:
            failure_reasons.append(f"word_count_exceeds_5 ({wc})")
            triggered_rules.append("word_count")
        if not placement_ok:
            failure_reasons.append("concluding_placement_violation")
            triggered_rules.append("placement")
        if repetition_flag:
            failure_reasons.append("lemma_repetition: " + ", ".join(repeated[:6]))
            triggered_rules.append("repetition")
        if thematic_ok is False:
            failure_reasons.append("thematic_cohesion_violation (next <= prev)")
            triggered_rules.append("thematic_cohesion")
        elif thematic_ok is None:
            if embed_model is None or not embeddings:
                failure_reasons.append("thematic_cohesion_unknown (embeddings missing)")
        final = "PASS" if not any(fr for fr in failure_reasons if not fr.startswith("thematic_cohesion_unknown")) else "FAIL"
        results.append(TransitionResult(
            article_id=r.get("article_id"),
            para_idx=r.get("para_idx"),
            transition_text=ttext,
            word_count=wc,
            is_concluding=is_conc,
            placement_check=placement_ok,
            repeated_lemmas=repeated,
            repetition_flag=repetition_flag,
            similarity_next=sim_next,
            similarity_prev=sim_prev,
            thematic_ok=thematic_ok,
            final_verdict=final,
            failure_reasons=failure_reasons,
            triggered_rules=triggered_rules
        ))
    return results

st.set_page_config(page_title="Transition QA (file-based)", layout="wide")
st.title("Prototype QA Tool — Transition QA for French news")

with st.sidebar:
    st.markdown("**Upload** article files (.txt, .docx) or a CSV with required columns.")
    st.text("CSV required columns: article_id, para_idx, transition_text, prev_paragraph, next_paragraph")
    model_name = st.text_input("Sentence-transformer model", value=DEFAULT_EMBEDDING_MODEL)
    thematic_delta = st.number_input("Thematic margin threshold (sim_next - sim_prev)", value=0.0, format="%.3f")
    attempt_spacy = st.checkbox("Attempt to load spaCy FR for lemmatization", value=True)
    require_para_idx = st.checkbox("Require para_idx mapping for placement (stricter)", value=False)
    st.markdown("---")
    st.markdown("Required python packages: sentence_transformers, scikit-learn, pandas. Optional: spacy (+fr model) and python-docx.")
    if not DOCX_AVAILABLE:
        st.markdown("Tip: to enable .docx parsing install `python-docx`.")
    st.markdown("Tip: provide CSV with reliable para_idx for best results.")

nlp = None
if attempt_spacy:
    model_ok = ensure_spacy_model("fr_core_news_sm")
    if model_ok:
        nlp = load_spacy_fr_model()
if attempt_spacy and nlp is None:
    st.sidebar.warning("spaCy FR model not found/loaded. The app will use a simple fallback lemmatizer. For best results, install a fr model (fr_core_news_sm).")

@st.cache_resource(show_spinner=False)
def load_embedding(name: str):
    try:
        return SentenceTransformer(name)
    except Exception as e:
        st.error(f"Failed to load embedding model '{name}': {e}")
        return None

with st.spinner("Loading embedding model (may take a few seconds)..."):
    embed_model = load_embedding(model_name)

if embed_model is None:
    st.sidebar.warning("No embedding model available — thematic checks will be marked unknown.")

uploaded_files = st.file_uploader("Upload files", type=["txt", "csv", "docx"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload article text files (.txt/.docx) or a CSV. The app will parse transitions from text files automatically or accept CSV with required columns.")
    st.stop()

all_rows = []
for up in uploaded_files:
    name = up.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Failed to read CSV {up.name}: {e}")
            continue
        required = {"article_id", "para_idx", "transition_text", "prev_paragraph", "next_paragraph"}
        if not required.issubset(df.columns):
            st.error(f"CSV {up.name} missing required columns: {required - set(df.columns)}")
            continue
        for _, r in df.iterrows():
            all_rows.append({
                "article_id": r.get("article_id"),
                "para_idx": int(r.get("para_idx")) if not pd.isna(r.get("para_idx")) else None,
                "transition_text": str(r.get("transition_text", "") or ""),
                "prev_paragraph": str(r.get("prev_paragraph", "") or ""),
                "next_paragraph": str(r.get("next_paragraph", "") or "")
            })
    else:
        txt = read_text_from_uploaded_file(up)
        paragraphs, transitions, metadata = extract_paragraphs_and_transitions_from_text(txt)
        article_id = up.name
        mapped = map_transitions_to_context(paragraphs, transitions)
        if not mapped:
            st.warning(f"No transitions extracted from {up.name}. Check file format.")
        for m in mapped:
            all_rows.append({
                "article_id": article_id,
                "para_idx": m.get("para_idx"),
                "transition_text": m.get("transition_text"),
                "prev_paragraph": m.get("prev_paragraph"),
                "next_paragraph": m.get("next_paragraph")
            })

if not all_rows:
    st.error("No transition rows to analyze after parsing uploads.")
    st.stop()

num_unmapped = sum(1 for r in all_rows if r.get("para_idx") is None)
if num_unmapped:
    st.warning(f"{num_unmapped} transition(s) had no para_idx mapping (may affect placement detection). Consider uploading CSV with para_idx.")

results_objs: List[TransitionResult] = []
for aid, group_rows in pd.DataFrame(all_rows).groupby("article_id", sort=False):
    rows_list = group_rows.to_dict(orient="records")
    res = analyze_group_rows(rows_list, nlp, embed_model, thematic_delta=thematic_delta, require_para_idx=require_para_idx)
    results_objs.extend(res)

def results_to_df(results: List[TransitionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        word_ok = r.word_count <= 5
        placement_ok = bool(r.placement_check)
        repetition_ok = not r.repetition_flag
        thematic_ok_bool = r.thematic_ok if r.thematic_ok is not None else np.nan
        rows.append({
            "article_id": r.article_id,
            "para_idx": r.para_idx,
            "transition_text": r.transition_text,
            "word_count": r.word_count,
            "word_ok": word_ok,
            "is_concluding": r.is_concluding,
            "placement_ok": placement_ok,
            "repetition_ok": repetition_ok,
            "repeated_lemmas": ", ".join(r.repeated_lemmas) if r.repeated_lemmas else "",
            "similarity_next": r.similarity_next if r.similarity_next is not None else np.nan,
            "similarity_prev": r.similarity_prev if r.similarity_prev is not None else np.nan,
            "thematic_ok": thematic_ok_bool,
            "final_verdict": r.final_verdict,
            "failure_reasons": "; ".join(r.failure_reasons) if r.failure_reasons else "",
            "triggered_rules": "; ".join(r.triggered_rules) if r.triggered_rules else ""
        })
    return pd.DataFrame(rows)

df_results = results_to_df(results_objs)

st.subheader("Results table")
show_only_fails = st.checkbox("Show only FAILs", value=False)
display_df = df_results[df_results["final_verdict"] == "FAIL"] if show_only_fails else df_results

st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

st.subheader("Summary")
total = len(df_results)
passed = (df_results["final_verdict"] == "PASS").sum()
failed = total - passed
c1, c2, c3 = st.columns(3)
c1.metric("Total transitions", total)
c2.metric("PASS", int(passed))
c3.metric("FAIL", int(failed))

st.markdown("**Most common failure reasons**")
fail_reasons: Dict[str, int] = {}
for fr in df_results["failure_reasons"].dropna():
    for part in str(fr).split(";"):
        p = part.strip()
        if p:
            fail_reasons[p] = fail_reasons.get(p, 0) + 1
st.write(pd.DataFrame(sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True), columns=["reason", "count"]))

csv_bytes = df_results.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv_bytes, file_name="transition_results.csv", mime="text/csv")

st.caption(
    "Parsing: For .txt files the app tries to extract the Article: section and a trailing 'Transitions' block. "
    "If transitions are inline between paragraphs (short lines), the parser will still detect them. "
    "If mapping can't be inferred, prev/next paragraphs will be empty for that transition. "
    "For reliable placement checks provide CSV with para_idx."
)

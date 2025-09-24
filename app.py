import os, re
import faiss
import numpy as np
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
)

# =============================
# Config
# =============================
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # optional

# Generator: keep T5 for speed on CPU
GENERATOR_ID = "google/flan-t5-base"
# If you later want nicer prose (slower on CPU), switch:
# GENERATOR_ID = "microsoft/Phi-3.5-mini-instruct"

CHUNK_CHARS     = 800
TOP_K           = 8
TOP_RERANK      = 3
SIM_THRESHOLD   = 0.25
MAX_NEW_TOKENS  = 160

# =============================
# Load models (CPU)
# =============================
embedder = SentenceTransformer(EMBEDDING_MODEL_ID)
reranker = CrossEncoder(RERANKER_ID)

# Auto-detect generator type (seq2seq vs causal)
gen_cfg = AutoConfig.from_pretrained(GENERATOR_ID)
tok = AutoTokenizer.from_pretrained(GENERATOR_ID)
IS_CAUSAL = (gen_cfg.is_encoder_decoder is False)

if IS_CAUSAL:
    gen = AutoModelForCausalLM.from_pretrained(GENERATOR_ID)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        gen.config.pad_token_id = tok.pad_token_id
else:
    gen = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_ID)

# =============================
# In-memory doc store
# =============================
class DocChunk:
    def __init__(self, doc_id, title, page, line_start, line_end, text):
        self.doc_id = doc_id
        self.title = title
        self.page = page
        self.line_start = line_start
        self.line_end = line_end
        self.text = text

INDEX = None
CHUNKS = []
DOC_TITLES = []

# =============================
# Helpers
# =============================
def _safe(msg: str) -> str:
    return f"**Note:** {msg}"

def _split_lines_keep_idx(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

def chunk_page_to_citable_segments(doc_id, title, page_num, page_text):
    lines = _split_lines_keep_idx(page_text)
    segments, buf, start_line = [], "", 0
    for i, ln in enumerate(lines):
        add = (ln + "\n")
        if not buf:
            start_line = i + 1
        if len(buf) + len(add) <= CHUNK_CHARS:
            buf += add
        else:
            end_line = i
            if buf.strip():
                segments.append(DocChunk(doc_id, title, page_num, start_line, end_line, buf.strip()))
            overlap = 3
            overlap_lines = lines[max(0, i - overlap):i]
            buf = ("\n".join(overlap_lines) + ("\n" if overlap_lines else "")) + add
            start_line = max(1, i - overlap) + 1
    if buf.strip():
        segments.append(DocChunk(doc_id, title, page_num, start_line, len(lines), buf.strip()))
    return segments

def build_index_from_pdfs(files):
    global INDEX, CHUNKS, DOC_TITLES
    CHUNKS, DOC_TITLES = [], []

    for doc_id, f in enumerate(files):
        title = os.path.splitext(os.path.basename(f.name))[0]
        DOC_TITLES.append(title)
        reader = PdfReader(f)
        for p_idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            text = re.sub(r"[ \t]+", " ", text)
            CHUNKS.extend(
                chunk_page_to_citable_segments(doc_id, title, p_idx + 1, text)
            )

    if not CHUNKS:
        return 0, 0

    texts = [c.text for c in CHUNKS]
    vecs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine (on normalized vectors)
    index.add(vecs.astype(np.float32))
    INDEX = index
    return len(DOC_TITLES), len(CHUNKS)

def search(query: str):
    if INDEX is None or not CHUNKS:
        return [], 0.0
    q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, idxs = INDEX.search(q.astype(np.float32), TOP_K)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()
    results = [(CHUNKS[i], float(scores[j])) for j, i in enumerate(idxs) if i != -1]
    best_score = max(scores) if scores else 0.0
    return results, best_score

def rerank(query, candidates):
    if not candidates:
        return []
    pairs = [[query, c.text] for c, _ in candidates]
    rr_scores = reranker.predict(pairs).tolist()
    ranked = sorted(
        [(c, s) for (c, _), s in zip(candidates, rr_scores)],
        key=lambda x: x[1],
        reverse=True,
    )
    return [c for c, _ in ranked[:TOP_RERANK]]

def make_citation_label(c: DocChunk, idx: int) -> str:
    return f"[{idx}] {c.title} p{c.page} L{c.line_start}-{c.line_end}"

def build_prompt(query, contexts):
    numbered = "\n\n".join([f"[{i+1}] {c.text.strip()}" for i, c in enumerate(contexts)]) or "(no context)"
    return (
        "You are an HR assistant. Based ONLY on the context snippets, "
        "write a clear, short answer (3–5 sentences). "
        "Do not output only citation numbers like [1]; write a natural explanation "
        "and include inline citations like [1], [2] when you use a snippet.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{numbered}\n\n"
        "Answer:"
    )

def generate_answer(query, retrieved, best_score):
    if best_score < SIM_THRESHOLD or not retrieved:
        previews = []
        for i, (chunk, _score) in enumerate(retrieved[:3], start=1):
            previews.append(f"{make_citation_label(chunk, i)}: {chunk.text[:240].strip()}...")
        if previews:
            return _safe("Couldn’t find an exact match. Closest snippets:\n\n" + "\n".join(previews))
        return _safe("No relevant snippets retrieved. Try a simpler query or upload more specific PDFs.")

    top_contexts = [c for (c, _s) in retrieved]
    if len(top_contexts) > 1:
        top_contexts = rerank(query, retrieved) or top_contexts[:TOP_RERANK]
    top_contexts = top_contexts[:TOP_RERANK]

    prompt = build_prompt(query, top_contexts)

    if IS_CAUSAL:
        inputs = tok(prompt, return_tensors="pt", truncation=True)
        out = gen.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=4,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    else:
        input_ids = tok(prompt, return_tensors="pt", truncation=True).input_ids
        out = gen.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            length_penalty=0.8,
            early_stopping=True,
        )

    text = tok.decode(out[0], skip_special_tokens=True).strip()
    if text in {"[1]", "[2]", "[3]", ""}:
        text = "Summary: " + top_contexts[0].text[:300].strip() + "… [1]"

    legend = [make_citation_label(c, i) for i, c in enumerate(top_contexts, start=1)]
    return text + "\n\nReferences:\n" + "\n".join(legend)

# =============================
# Gradio UI
# =============================
with gr.Blocks(title="HR Policies – Document Q&A") as demo:
    gr.Markdown(
        "# 🧭 HR Policies – Document Q&A\n"
        "Upload your HR policy PDFs and ask questions. The app retrieves relevant sections and "
        "generates a short answer with inline citations.\n"
        "**Private:** documents are processed in-memory for this session.\n"
    )

    with gr.Row():
        files = gr.File(label="Upload HR Policy PDFs", file_types=[".pdf"], file_count="multiple")
        build_btn = gr.Button("Build index")

    info = gr.Markdown("No documents indexed yet.")

    def _build(files):
        if not files:
            return "Please upload at least one PDF."
        n_docs, n_chunks = build_index_from_pdfs(files)
        return f"Indexed **{n_docs}** document(s) into **{n_chunks}** chunks. Ready for questions."

    build_btn.click(fn=_build, inputs=[files], outputs=[info])

    question = gr.Textbox(
        label="Ask a question (e.g., 'How many casual leave days are allowed during probation?')",
        lines=2,
    )
    ask_btn = gr.Button("Answer")
    answer = gr.Markdown(label="Answer")

    def _answer(q: str):
        """
        Answer a user question with retrieval -> (optional) rerank -> generate.
        Shows a small progress toast so the UI doesn't feel stuck.
        """
        try:
            q = (q or "").strip()
            if not q:
                return _safe("Please enter a question.")
            if INDEX is None or not CHUNKS:
                return _safe("No index found. Upload a PDF and click **Build index** first.")

            gr.Info("Finding relevant policy text…")
            candidates, best = search(q)

            if not candidates:
                return _safe("No snippets retrieved. Try a simpler question or upload another PDF.")

            if best < SIM_THRESHOLD:
                previews = []
                for i, (chunk, score) in enumerate(candidates[:3], start=1):
                    previews.append(f"- {make_citation_label(chunk, i)} (sim={score:.2f}): {chunk.text[:180]}…")
                return _safe("Couldn’t find an exact match. Closest snippets:\n\n" + "\n".join(previews))

            gr.Info("Generating a short answer with citations…")
            return generate_answer(q, candidates, best)

        except Exception as e:
            print("Answer error:", e)
            return _safe(f"An error occurred while answering: `{type(e).__name__}: {e}`. Check **Logs**.")

    ask_btn.click(fn=_answer, inputs=[question], outputs=[answer])

    gr.Markdown(
        "### Tips\n"
        "- Keep PDFs focused (e.g., *Leave Policy.pdf*, *Travel Policy.pdf*).\n"
        "- If answers fall back to ‘no exact match’, slightly lower the similarity threshold (code: `SIM_THRESHOLD`).\n"
        "- For longer answers, increase `MAX_NEW_TOKENS`.\n"
        f"- Current generator: `{GENERATOR_ID}` (causal={IS_CAUSAL}).\n"
    )

if __name__ == "__main__":
    demo.launch()

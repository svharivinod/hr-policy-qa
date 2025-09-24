# 📄 HR Policy Q&A – Instant Answers from Policy Documents

Employees often struggle to find specific details hidden in long HR policy PDFs — like “How many casual leaves are allowed during probation?” or “What’s the reimbursement rule for travel?”.

This project solves that problem with an AI-powered Document Q&A app built using Gradio + Hugging Face.
Upload your HR policy documents and simply ask questions in plain English. The app retrieves the most relevant snippets and generates a short, clear answer with inline citations so you know exactly where the rule comes from.

# 🚀 Features

📂 Upload PDFs – Supports multiple HR policy documents.

🔍 Semantic Search – Finds the most relevant sections using embeddings.

🏷️ Inline Citations – Each answer is backed by the actual source (with page & line numbers).

🛡️ Guardrails – If the system isn’t confident, it won’t hallucinate; it shows closest snippets instead.

⚡ Runs on CPU – No GPU required, fully deployable on Hugging Face Spaces.

# ⚙️ Tech Stack

1) Embeddings: BAAI/bge-small-en-v1.5

2) Vector Store: FAISS (fast similarity search)

3) Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

4) Generator: google/flan-t5-base (CPU-friendly)

5) Interface: Gradio + Hugging Face Spaces

# 📷 Demo
<img width="1913" height="684" alt="HR_Document_QA" src="https://github.com/user-attachments/assets/8bb90529-6580-4683-8efb-4559c77a24da" />

# 🛠️ Setup

Clone the repo:
git clone https://github.com/<your-username>/hr-policy-qa.git
cd hr-policy-qa
pip install -r requirements.txt
python app.py
The app will launch on http://127.0.0.1:7860

# 💡 Business Use Cases

HR Policy Documents

Company Handbooks

Compliance Manuals

Legal Documents

Customer Support Knowledge Base

Anywhere you have long documents and repetitive queries, this approach can be applied.

## ✨ If you found this helpful, consider giving the repo a ⭐ and trying it out with your own documents!





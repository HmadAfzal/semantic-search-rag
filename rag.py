from pypdf import PdfReader
import re
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()
model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def load_pdf(path):
  reader = PdfReader(path)
  text=""
  for page in reader.pages:
    text+=page.extract_text()
  return text

def chunk_text(text, chunk_size=500, overlap_sentences=2):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    current_sentences = []

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
            current_sentences.append(sentence)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap = current_sentences[-overlap_sentences:]
            current_chunk = " ".join(overlap) + " " + sentence + " "
            current_sentences = overlap + [sentence]

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def embed_chunks(chunks):
  embeded_chunks=model.encode(chunks)
  embeded_chunks=np.array(embeded_chunks, dtype="float32")
  return embeded_chunks

def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def FindIndex(index, question, k):
  embeded_chunk=embed_chunks(question)
  embeded_chunk=embeded_chunk.reshape(1,-1)
  D,I=index.search(embeded_chunk, k)
  return D,I


def generate_answer(question, chunks, retrieved_indices):
    context = ""
    for i in retrieved_indices[0]:
        context += chunks[i] + "\n\n"

    prompt = f"""<|system|>
You are an expert assistant that answers questions strictly based on the provided context.
Follow these rules:
- If the answer is in the context, answer clearly and concisely in 2-3 sentences.
- If the answer is NOT in the context, say exactly "I don't have enough information to answer this question."
- Never make up information.
- Always be specific and precise.</s>
<|user|>
Context:
{context}

Question: {question}</s>
<|assistant|>"""
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=400
    )
    answer = response.choices[0].message.content
    return answer


def rag_pipeline(pdf_path, question, k=3):
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = build_index(embeddings)
    D,I = FindIndex(index, question, k)
    answer = generate_answer(question, chunks, I)
    return answer

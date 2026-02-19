
import os
import sys
import json
import re
import time
import numpy as np
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bm25s

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True

    vllm_clients = {
        'qwen': OpenAI(api_key="EMPTY", base_url="http://0.0.0.0:8000/v1"),
        'gpt-oss': OpenAI(api_key="EMPTY", base_url="http://0.0.0.0:8000/v1"),
    }
    VLLM_AVAILABLE = True

    openai_api_key = os.environ.get('OPENAI_API_KEY', '')
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    else:
        openai_client = None

except ImportError:
    OPENAI_AVAILABLE = False
    VLLM_AVAILABLE = False
    vllm_clients = {}
    openai_client = None
    try:
        from ollama import chat
    except ImportError:
        pass

os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

try:
    from transformers import AutoTokenizer, AutoModel
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except Exception:
    CROSSENCODER_AVAILABLE = False


# ── Model caching ──

_fenice_model = None
_fenice_tokenizer = None
_gemma_pipeline = None


def get_fenice_model():
    global _fenice_model, _fenice_tokenizer
    if _fenice_model is None:
        if not torch.cuda.is_available():
            raise SystemError("GPU is required to run this code.")
        device = torch.device("cuda:0")
        _fenice_tokenizer = T5Tokenizer.from_pretrained("Babelscape/t5-base-summarization-claim-extractor")
        _fenice_model = T5ForConditionalGeneration.from_pretrained("Babelscape/t5-base-summarization-claim-extractor")
        _fenice_model.to(device)
    return _fenice_model, _fenice_tokenizer


def get_gemma_pipeline():
    global _gemma_pipeline
    if _gemma_pipeline is None:
        from transformers import pipeline, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        _gemma_pipeline = pipeline(
            "text-generation",
            model="google/gemma-7b-it",
            model_kwargs={"quantization_config": bnb_config},
            device_map={"": 0},
        )
    return _gemma_pipeline


# ── Claim extraction ──

def extract_claims_from_review_fenice(review):
    model, tokenizer = get_fenice_model()
    device = torch.device("cuda")
    tok_input = tokenizer.batch_encode_plus([review], return_tensors="pt", padding=True)
    tok_input = {key: value.to(device) for key, value in tok_input.items()}
    claims = model.generate(**tok_input)
    claims = tokenizer.batch_decode(claims, skip_special_tokens=True)
    all_claims = []
    for claim_text in claims:
        if claim_text.strip():
            all_claims.extend([line.strip() for line in claim_text.split('\n') if line.strip()])
    return all_claims


def extract_claims_from_review_gemma(review):
    try:
        pipe = get_gemma_pipeline()
        prompt = f"Extract the main claims from the following review text. List each claim on a new line:\n\nReview: {review}\n\nClaims:"
        result = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
        claims_text = result[0]['generated_text'].split("Claims:")[-1].strip()
        return [c.strip() for c in claims_text.split('\n') if c.strip()]
    except Exception as e:
        print(f"Error with Gemma extraction: {e}")
        sentences = review.split('.')
        return [s.strip() for s in sentences[:3] if s.strip()]


# ── Helper to prepare chunks ──

def _prepare_chunks(chunks):
    chunk_texts = []
    valid_chunks = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            chunk_dict = chunk.copy()
        else:
            text = str(chunk)
            chunk_dict = {"text": text, "section": "unknown"}
        if text.strip():
            chunk_texts.append(text)
            if 'section' not in chunk_dict:
                chunk_dict['section'] = 'unknown'
            valid_chunks.append(chunk_dict)
    return chunk_texts, valid_chunks


# ── Retrieval functions ──

def retrieve_top_k_evidences_tfidf(query: str, chunks: list, top_k: int = 3) -> list:
    if not chunks:
        return []
    chunk_texts, valid_chunks = _prepare_chunks(chunks)
    if not chunk_texts:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(chunk_texts)
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [valid_chunks[idx] for idx in top_indices][:top_k]
    except Exception as e:
        print(f"Error in TF-IDF retrieval: {e}")
        return []


def retrieve_top_k_evidences_bm25(query: str, chunks: list, top_k: int = 3) -> list:
    if not chunks:
        return []
    chunk_texts, valid_chunks = _prepare_chunks(chunks)
    if not chunk_texts:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(chunk_texts)
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [valid_chunks[idx] for idx in top_indices][:top_k]
    except Exception as e:
        print(f"Error in BM25 retrieval: {e}")
        return []


def retrieve_top_k_evidences_sbert(query: str, chunks: list, top_k: int = 3) -> list:
    if not SBERT_AVAILABLE or not chunks:
        return []
    chunk_texts, valid_chunks = _prepare_chunks(chunks)
    if not chunk_texts:
        return []
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        query_tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
        with torch.no_grad():
            query_embedding = model(**query_tokens).last_hidden_state.mean(dim=1)

        similarities = []
        for chunk_text in chunk_texts:
            chunk_tokens = tokenizer(chunk_text, return_tensors="pt", padding=True, truncation=True)
            chunk_tokens = {k: v.to(device) for k, v in chunk_tokens.items()}
            with torch.no_grad():
                chunk_embedding = model(**chunk_tokens).last_hidden_state.mean(dim=1)
                similarities.append(torch.cosine_similarity(query_embedding, chunk_embedding).item())

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [valid_chunks[idx] for idx in top_indices][:top_k]
    except Exception as e:
        print(f"Error in SBERT retrieval: {e}")
        return []


def retrieve_top_k_evidences_faiss(query: str, chunks: list, top_k: int = 3) -> list:
    try:
        import faiss
    except ImportError:
        return retrieve_top_k_evidences_sbert(query, chunks, top_k)

    if not SBERT_AVAILABLE or not chunks:
        return []
    chunk_texts, valid_chunks = _prepare_chunks(chunks)
    if not chunk_texts:
        return []
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        chunk_embeddings = []
        for chunk_text in chunk_texts:
            chunk_tokens = tokenizer(chunk_text, return_tensors="pt", padding=True, truncation=True)
            chunk_tokens = {k: v.to(device) for k, v in chunk_tokens.items()}
            with torch.no_grad():
                emb = model(**chunk_tokens).last_hidden_state.mean(dim=1)
                chunk_embeddings.append(emb.cpu().numpy())

        chunk_embeddings = np.vstack(chunk_embeddings)
        embedding_dim = chunk_embeddings.shape[1]

        index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(chunk_embeddings)
        index.add(chunk_embeddings)

        query_tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
        with torch.no_grad():
            query_embedding = model(**query_tokens).last_hidden_state.mean(dim=1)
        query_embedding_np = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding_np)

        _, top_indices = index.search(query_embedding_np, min(top_k, len(chunk_texts)))
        return [valid_chunks[idx] for idx in top_indices[0][:top_k]]
    except Exception as e:
        print(f"Error in FAISS retrieval: {e}")
        return []


def retrieve_top_k_evidences_biencoder_crossencoder(
    query: str, chunks: list, top_k: int = 3, initial_k: int = 20,
    bi_encoder_model: str = 'BAAI/bge-large-en-v1.5',
    cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
) -> list:
    if not SBERT_AVAILABLE or not chunks:
        return []
    chunk_texts, valid_chunks = _prepare_chunks(chunks)
    if not chunk_texts:
        return []

    initial_k = min(initial_k, len(chunk_texts))
    top_k = min(top_k, initial_k)

    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        bi_tokenizer = AutoTokenizer.from_pretrained(bi_encoder_model)
        bi_model = AutoModel.from_pretrained(bi_encoder_model, low_cpu_mem_usage=False)
        bi_model = bi_model.to(device)
        bi_model.eval()

        def get_embeddings(texts, instruction_prefix=""):
            embeddings = []
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                if 'e5' in bi_encoder_model.lower():
                    batch = [instruction_prefix + t for t in batch]
                inputs = bi_tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = bi_model(**inputs)
                    mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    emb = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    embeddings.append(emb)
            return torch.cat(embeddings, dim=0)

        q_prefix = "query: " if 'e5' in bi_encoder_model.lower() else ""
        p_prefix = "passage: " if 'e5' in bi_encoder_model.lower() else ""
        query_emb = get_embeddings([query], instruction_prefix=q_prefix)
        chunk_emb = get_embeddings(chunk_texts, instruction_prefix=p_prefix)

        similarities = torch.matmul(query_emb, chunk_emb.T).squeeze(0)
        top_initial_indices = torch.argsort(similarities, descending=True)[:initial_k]
        top_initial_chunks = [valid_chunks[i.item()] for i in top_initial_indices]
        top_initial_texts = [chunk_texts[i.item()] for i in top_initial_indices]

        if not CROSSENCODER_AVAILABLE:
            return top_initial_chunks[:top_k]

        cross_encoder = CrossEncoder(cross_encoder_model, max_length=512, device=device)
        pairs = [[query, text] for text in top_initial_texts]
        cross_scores = cross_encoder.predict(pairs, show_progress_bar=False)
        ranked_indices = np.argsort(cross_scores)[::-1]
        return [top_initial_chunks[i] for i in ranked_indices[:top_k]]

    except Exception as e:
        print(f"Error in Bi-encoder + Cross-encoder retrieval: {e}")
        import traceback
        traceback.print_exc()
        return []


def retrieve_top_k_evidences_rrf(query: str, chunks: list, top_k: int = 3, initial_k: int = 20, k: int = 60) -> list:
    if not chunks:
        return []
    chunk_texts, valid_chunks = _prepare_chunks(chunks)
    if not chunk_texts:
        return []
    try:
        initial_k = min(initial_k, len(chunk_texts))

        bm25_ranks = {}
        try:
            retriever = bm25s.BM25(corpus=chunk_texts)
            tokenized_corpus = bm25s.tokenize(chunk_texts)
            retriever.index(tokenized_corpus)
            tokenized_query = bm25s.tokenize(query)
            results, scores = retriever.retrieve(tokenized_query, k=initial_k)
            for rank, result_text in enumerate(results[0]):
                for idx, ct in enumerate(chunk_texts):
                    if ct == result_text:
                        bm25_ranks[idx] = rank
                        break
        except Exception:
            pass

        semantic_ranks = {}
        if SBERT_AVAILABLE:
            try:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                model.eval()

                qt = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
                qt = {kk: v.to(device) for kk, v in qt.items()}
                with torch.no_grad():
                    qe = model(**qt).last_hidden_state.mean(dim=1)

                sims = []
                for ct in chunk_texts:
                    ct_tok = tokenizer(ct, return_tensors="pt", padding=True, truncation=True)
                    ct_tok = {kk: v.to(device) for kk, v in ct_tok.items()}
                    with torch.no_grad():
                        ce = model(**ct_tok).last_hidden_state.mean(dim=1)
                        sims.append(torch.cosine_similarity(qe, ce).item())

                for rank, idx in enumerate(np.argsort(sims)[::-1][:initial_k]):
                    semantic_ranks[idx] = rank
            except Exception:
                pass

        rrf_scores = {}
        for idx in range(len(chunk_texts)):
            score = 0.0
            if idx in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[idx])
            if idx in semantic_ranks:
                score += 1.0 / (k + semantic_ranks[idx])
            if score > 0:
                rrf_scores[idx] = score

        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [valid_chunks[idx] for idx in sorted_indices[:top_k]]

    except Exception as e:
        print(f"Error in RRF retrieval: {e}")
        return retrieve_top_k_evidences_bm25(query, chunks, top_k)


def retrieve_top_k_evidences_bm25_crossencoder(query: str, chunks: list, top_k: int = 3, initial_k: int = 20) -> list:
    if not chunks:
        return []
    chunk_texts, valid_chunks = _prepare_chunks(chunks)
    if not chunk_texts:
        return []
    try:
        initial_k = min(initial_k, len(chunk_texts))

        retriever = bm25s.BM25(corpus=chunk_texts)
        tokenized_corpus = bm25s.tokenize(chunk_texts)
        retriever.index(tokenized_corpus)
        tokenized_query = bm25s.tokenize(query)
        results, scores = retriever.retrieve(tokenized_query, k=initial_k)

        top_candidates = []
        top_candidate_texts = []
        for result_text in results[0]:
            for idx, ct in enumerate(chunk_texts):
                if ct == result_text:
                    top_candidates.append(valid_chunks[idx])
                    top_candidate_texts.append(ct)
                    break

        if CROSSENCODER_AVAILABLE and len(top_candidates) > top_k:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device=device)
            pairs = [[query, text] for text in top_candidate_texts]
            cross_scores = cross_encoder.predict(pairs, show_progress_bar=False)
            ranked_indices = np.argsort(cross_scores)[::-1]
            return [top_candidates[i] for i in ranked_indices[:top_k]]
        else:
            return top_candidates[:top_k]

    except Exception as e:
        print(f"Error in BM25+CrossEncoder retrieval: {e}")
        return retrieve_top_k_evidences_bm25(query, chunks, top_k)


def load_chunks_for_submission(submission_id: str, base_data_path: str) -> list:
    try:
        chunks_path = os.path.join(base_data_path, "4_chunked", "docling-standard", f"{submission_id}_chunks.jsonl")
        if not os.path.exists(chunks_path):
            return []
        chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'chunks' in data:
                        chunks.extend(data['chunks'])
        return chunks
    except Exception as e:
        print(f"Error loading chunks for {submission_id}: {e}")
        return []


# ── Claim verification ──

def entailment_for_claim(claim: str, evidences: list, model: str) -> dict:
    evidence_texts = []
    evidence_sections = []
    for ev in evidences:
        if isinstance(ev, dict):
            evidence_texts.append(ev.get("text", str(ev)))
            evidence_sections.append(ev.get("section", "unknown"))
        else:
            evidence_texts.append(str(ev))
            evidence_sections.append("unknown")

    prompt = (
        "You are a factual-verification API. Respond **ONLY** with a valid JSON object -- no other text.\n\n"
        "Label definitions you must use:\n"
        "  - Supported: The evidence fully backs the claim with no gaps or contradictions.\n"
        "  - Partially Supported: Some parts align, but other details are missing or unclear.\n"
        "  - Contradicted: The claim directly conflicts with the evidence.\n"
        "  - Not Determined: The evidence is insufficient to confirm or deny the claim.\n\n"
        "Task: Classify the claim using exactly one of the four labels above.\n\n"
        f"CLAIM:\n{claim}\n\n"
        "EVIDENCE SNIPPETS (each separated by '---'):\n"
        + "\n---\n".join(evidence_texts)
        + "\n\n"
        "Return exactly:\n"
        "{\n"
        '  "result": "<Supported|Partially Supported|Contradicted|Not Determined>",\n'
        '  "justification": "<brief explanation>"\n'
        "}"
    )

    try:
        is_openai_model = (
            model.startswith('gpt-') or model.startswith('o3-') or model.startswith('o4-')
            or model in ['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
        )

        if is_openai_model and openai_client:
            response = openai_client.chat.completions.create(
                model=model, messages=[{'role': 'user', 'content': prompt}],
            )
            response_text = response.choices[0].message.content.strip()
        elif VLLM_AVAILABLE:
            if 'gpt-oss' in model.lower() or '20b' in model.lower():
                client = vllm_clients.get('gpt-oss', vllm_clients.get('qwen'))
            else:
                client = vllm_clients.get('qwen', list(vllm_clients.values())[0] if vllm_clients else None)
            if client:
                response = client.chat.completions.create(
                    model=model, messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0, max_tokens=512, top_p=0.9, seed=42,
                )
                response_text = response.choices[0].message.content.strip()
            else:
                raise Exception("No vLLM client available")
        else:
            response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
            response_text = response['message']['content'].strip()

        try:
            result = json.loads(response_text)
            result['evidence_sections'] = evidence_sections
            return result
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result['evidence_sections'] = evidence_sections
                return result
            return {'result': 'Undetermined', 'justification': f'Failed to parse: {response_text[:100]}', 'evidence_sections': evidence_sections}

    except Exception as e:
        return {'result': 'Undetermined', 'justification': f'Error: {str(e)}', 'evidence_sections': evidence_sections}

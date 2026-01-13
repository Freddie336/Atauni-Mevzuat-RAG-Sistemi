import os
import streamlit as st
import logging
import time
import hashlib
import json
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# Config dosyasını import et
import sys
from pathlib import Path

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.config import (
        DB_FAISS_PATH, EMBEDDING_MODEL, RETRIEVER_K, RETRIEVER_FETCH_K,
        RETRIEVER_LAMBDA_MULT, RETRIEVER_SEARCH_TYPE, MAX_CONTEXT_TOKENS,
        TOKEN_ESTIMATION_FACTOR, LLM_MODEL, TEMPERATURE_RAG,
        MAX_QUERY_LENGTH, MIN_QUERY_LENGTH, USE_RERANKING, RERANKER_MODEL, RERANK_TOP_K,
        USE_HYBRID_SEARCH, HYBRID_SEMANTIC_K, HYBRID_KEYWORD_K, HYBRID_FINAL_K,
        USE_RETRIEVAL_CACHE, RETRIEVAL_CACHE_SIZE, TRACK_PERFORMANCE, PERFORMANCE_LOG_FILE,
        USE_INLINE_CITATIONS, CITATION_THRESHOLD, MAX_CITATIONS_PER_SENTENCE
    )
except ImportError:
    DB_FAISS_PATH = "vectorstore/db_faiss"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RETRIEVER_K = 10
    RETRIEVER_FETCH_K = 20
    RETRIEVER_LAMBDA_MULT = 0.7
    RETRIEVER_SEARCH_TYPE = "mmr"
    MAX_CONTEXT_TOKENS = 4000
    TOKEN_ESTIMATION_FACTOR = 1.3
    LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
    TEMPERATURE_RAG = 0.2
    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 3
    USE_RERANKING = True
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K = 10
    USE_HYBRID_SEARCH = True
    HYBRID_SEMANTIC_K = 15
    HYBRID_KEYWORD_K = 15
    HYBRID_FINAL_K = 10
    USE_RETRIEVAL_CACHE = True
    RETRIEVAL_CACHE_SIZE = 100
    TRACK_PERFORMANCE = True
    PERFORMANCE_LOG_FILE = "performance_logs.json"
    USE_INLINE_CITATIONS = False
    CITATION_THRESHOLD = 0.5
    MAX_CITATIONS_PER_SENTENCE = 2

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== MODERN UI CSS ====================
def load_custom_css():
    st.markdown("""
    <style>
    /* Ana container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Chat container */
    .stChatMessage {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.3s ease-in;
        color:#1a1a1a;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* User message */
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #1a1a1a;
        border-left: 5px solid #FFD700;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 5px solid #667eea;
        color:#1a1a1a;
    }
    
    /* Input box */
    .stChatInputContainer {
        background: white;
        border-radius: 25px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #FFD700;
    }
    
    .stError {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #FFD700;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #FFD700;
    }
    
    /* Header */
    h1 {
        color: white;
        text-align: center;
        font-size: 3em;
        font-weight: 800;
        text-shadow: 0px 2px 8px rgba(255, 255, 255, 0.3), 
                    0px 4px 12px rgba(102, 126, 234, 0.5);
        margin-bottom: 10px;
        animation: slideDown 0.5s ease-out;
        letter-spacing: 2px;
    }

    /* Anchor link gizle */
    h1 a, .anchor-link {
        display: none !important;
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Subtitle */
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 30px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.2);
        padding: 15px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #FFD700 !important;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #FFD700;
    }
    
    /* Markdown in messages */
    .stMarkdown {
        line-height: 1.6;
        color:#1a1a1a;
    }
    
    /* Links */
    a {
        color: #FFD700;
        text-decoration: none;
        font-weight: bold;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Chat input placeholder */
    .stChatInput input::placeholder {
        color: #999;
        font-style: italic;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== UTILITY FUNCTIONS ====================
@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Optional[FAISS]:
    """FAISS vector store'u yükler ve cache'ler."""
    try:
        db_path = Path(DB_FAISS_PATH).resolve()
        
        if not db_path.exists():
            logger.warning(f"Config path bulunamadı: {db_path}, project_root kullanılıyor")
            db_path = project_root / "vectorstore" / "db_faiss"
            db_path = db_path.resolve()
        
        if not db_path.exists():
            cwd_path = Path.cwd() / "vectorstore" / "db_faiss"
            if cwd_path.exists():
                db_path = cwd_path.resolve()
                logger.info(f"Vector store CWD'den bulundu: {db_path}")
            else:
                script_path = Path(__file__).parent.parent / "vectorstore" / "db_faiss"
                if script_path.exists():
                    db_path = script_path.resolve()
                    logger.info(f"Vector store script dizininden bulundu: {db_path}")
        
        index_faiss = db_path / "index.faiss"
        index_pkl = db_path / "index.pkl"
        
        logger.info(f"Vector store yükleniyor: {db_path}")
        
        if not db_path.exists():
            error_msg = f"Vector store klasörü bulunamadı: {db_path}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        
        if not index_faiss.exists():
            error_msg = f"index.faiss dosyası bulunamadı: {index_faiss}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        
        if not index_pkl.exists():
            error_msg = f"index.pkl dosyası bulunamadı: {index_pkl}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        normalized_path = str(db_path.resolve())
        logger.info(f"FAISS load_local çağrılıyor: {normalized_path}")
        db = FAISS.load_local(normalized_path, embedding_model, allow_dangerous_deserialization=True)
        logger.info("Vector store başarıyla yüklendi")
        return db
    except FileNotFoundError as e:
        error_msg = f"Vector store bulunamadı: {DB_FAISS_PATH}. Hata: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Vector store yüklenemedi: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


@st.cache_resource
def get_llm() -> ChatGroq:
    """LLM'i yükler ve cache'ler."""
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable bulunamadı")
        
        logger.info(f"LLM yükleniyor: {LLM_MODEL}")
        llm = ChatGroq(
            model_name=LLM_MODEL,
            temperature=TEMPERATURE_RAG,
            groq_api_key=api_key,
        )
        logger.info("LLM başarıyla yüklendi")
        return llm
    except Exception as e:
        logger.error(f"LLM yüklenemedi: {str(e)}", exc_info=True)
        raise

def estimate_tokens(text: str) -> int:
    """Yaklaşık token sayısını hesaplar."""
    return int(len(text.split()) * TOKEN_ESTIMATION_FACTOR)


def limit_context_size(docs: List, max_tokens: int = MAX_CONTEXT_TOKENS) -> List:
    """Context'i token limitine göre sınırlar."""
    total_tokens = 0
    limited_docs = []
    
    for doc in docs:
        doc_tokens = estimate_tokens(doc.page_content)
        if total_tokens + doc_tokens > max_tokens:
            logger.warning(f"Token limiti aşıldı. {len(limited_docs)} doküman kullanılıyor")
            break
        limited_docs.append(doc)
        total_tokens += doc_tokens
    
    return limited_docs


def extract_keywords(text: str) -> set:
    """Metinden anahtar kelimeleri çıkarır."""
    import re
    
    stop_words = {
        'bir', 'bu', 'şu', 'o', 've', 'ile', 'için', 'gibi', 'kadar', 'daha',
        'en', 'çok', 'az', 'var', 'yok', 'ise', 'olan', 'olarak', 'ki', 'de',
        'da', 'den', 'dan', 'nin', 'nın', 'nun', 'nün', 'ler', 'lar',
        'leri', 'ları', 'in', 'ın', 'un', 'ün', 'e', 'a', 'i', 'ı', 'u', 'ü',
        'den', 'dan', 'ten', 'tan', 'ile', 'ce', 'ca', 'çe', 'ça', 'mi', 'mı',
        'mu', 'mü', 'dir', 'dır', 'dur', 'dür', 'tir', 'tır', 'tur', 'tür'
    }
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    
    keywords = {word for word in words if len(word) >= 3 and word not in stop_words}
    
    return keywords


def check_hallucination(response: str, source_docs: List) -> Dict:
    """Halüsinasyon kontrolü yapar."""
    if not response or not source_docs:
        return {
            'is_hallucinated': True,
            'confidence': 0.0,
            'match_ratio': 0.0,
            'warnings': ['Cevap veya kaynak bulunamadı'],
            'matched_keywords': [],
            'unmatched_keywords': []
        }
    
    response_keywords = extract_keywords(response)
    
    if not response_keywords:
        return {
            'is_hallucinated': False,
            'confidence': 1.0,
            'match_ratio': 1.0,
            'warnings': [],
            'matched_keywords': [],
            'unmatched_keywords': []
        }
    
    source_text = ' '.join([doc.page_content for doc in source_docs])
    source_keywords = extract_keywords(source_text)
    
    general_terms = {
        'madde', 'yönetmelik', 'yönerge', 'kural', 'düzenleme', 'prosedür',
        'kategori', 'sınıf', 'tür', 'genel', 'örnek', 'gibi', 'şekilde'
    }
    
    specific_terms = response_keywords - general_terms
    matched_keywords = response_keywords & source_keywords
    unmatched_keywords = response_keywords - source_keywords
    
    if specific_terms:
        matched_specific = specific_terms & source_keywords
        specific_match_ratio = len(matched_specific) / len(specific_terms)
    else:
        specific_match_ratio = 0.0
    
    match_ratio = len(matched_keywords) / len(response_keywords) if response_keywords else 0.0
    
    if specific_terms:
        confidence = (specific_match_ratio * 0.7) + (match_ratio * 0.3)
    else:
        confidence = match_ratio
    
    is_hallucinated = False
    if specific_terms:
        is_hallucinated = specific_match_ratio < 0.4
    else:
        is_hallucinated = match_ratio < 0.2
    
    warnings = []
    if is_hallucinated:
        warnings.append(f"⚠️ Bu cevap yönetmeliklerde tam olarak bulunmayan bilgiler içerebilir")
    elif match_ratio < 0.5:
        warnings.append(f"ℹ️ Bu cevap kısmen yönetmeliklere dayanıyor")
    
    logger.info(f"Halüsinasyon kontrolü: confidence={confidence:.2f}, is_hallucinated={is_hallucinated}")
    
    return {
        'is_hallucinated': is_hallucinated,
        'confidence': confidence,
        'match_ratio': match_ratio,
        'warnings': warnings,
        'matched_keywords': list(matched_keywords)[:10],
        'unmatched_keywords': list(unmatched_keywords)[:10]
    }


def is_university_question(query: str) -> bool:
    """Sorunun üniversite ile ilgili olup olmadığını kontrol eder."""
    query_lower = query.lower()
    
    greetings = ['merhaba', 'selam', 'iyi günler', 'günaydın', 'iyi akşamlar', 
                 'nasılsın', 'naber', 'hi', 'hello', 'hey', 'iyi geceler']
    if any(greeting in query_lower for greeting in greetings):
        return True
    
    non_university_keywords = [
        'tarif', 'yemek', 'pasta', 'kek', 'tatlı', 'çorba',
        'futbol', 'basketbol', 'maç', 'film', 'dizi', 'müzik',
        'hastalık', 'tedavi', 'ilaç', 'doktor', 'hastane'
    ]
    
    has_non_university = any(keyword in query_lower for keyword in non_university_keywords)
    
    university_keywords = [
        'üniversite', 'fakülte', 'bölüm', 'öğrenci', 'akademik', 'ders',
        'sınav', 'not', 'kredi', 'akts', 'diploma', 'mezuniyet', 'kayıt',
        'yönetmelik', 'yönerge', 'senato', 'rektör', 'dekan', 'başkan',
        'yarıyıl', 'dönem', 'staj', 'tez', 'proje', 'ödev', 'devamsızlık',
        'ders programı', 'müfredat', 'öğretim üyesi', 'asistan', 'hoca',
        'danışman', 'lisans', 'yüksek lisans', 'doktora', 'önlisans',
        'kütüphane', 'kampüs', 'yurt', 'burs', 'harç', 'ücret',
        'disiplin', 'ceza', 'izin', 'mazeretimdir', 'başvuru'
    ]
    
    has_university = any(keyword in query_lower for keyword in university_keywords)
    
    if has_non_university and not has_university:
        return False
    
    return True


def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    """Sorgu doğrulaması yapar."""
    if not query or len(query.strip()) == 0:
        return False, "Lütfen bir soru girin!"
    
    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Soru çok uzun! Maksimum {MAX_QUERY_LENGTH} karakter."
    
    if len(query.strip()) < MIN_QUERY_LENGTH:
        return False, f"Soru çok kısa! Minimum {MIN_QUERY_LENGTH} karakter."
    
    dangerous_patterns = ['<script', 'javascript:', 'onerror=', 'onload=', 'eval(']
    query_lower = query.lower()
    for pattern in dangerous_patterns:
        if pattern in query_lower:
            return False, "Geçersiz karakterler içeren soru!"
    
    if not is_university_question(query):
        return False, "Üzgünüm, ben sadece üniversite yönetmelikleri hakkında yardımcı olabilirim."
    
    return True, None


def rerank_documents(query: str, documents: List, top_k: int = RERANK_TOP_K) -> List:
    """Dokümanları yeniden sıralar."""
    if not USE_RERANKING or not documents:
        return documents[:top_k] if documents else []
    
    try:
        from sentence_transformers import CrossEncoder
        
        logger.info(f"Re-ranking yapılıyor: {len(documents)} doküman")
        
        @st.cache_resource
        def load_reranker():
            return CrossEncoder(RERANKER_MODEL)
        
        reranker = load_reranker()
        
        pairs = [[query, doc.page_content[:500]] for doc in documents]
        scores = reranker.predict(pairs)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        relevance_threshold = -2.0
        filtered_indices = [i for i in ranked_indices if scores[i] > relevance_threshold]
        reranked_docs = [documents[i] for i in filtered_indices[:top_k]]
        
        if not filtered_indices:
            reranked_docs = [documents[i] for i in ranked_indices[:top_k]]
        
        return reranked_docs
    
    except ImportError:
        logger.warning("sentence-transformers bulunamadı")
        return documents[:top_k] if documents else []
    except Exception as e:
        logger.error(f"Re-ranking hatası: {str(e)}", exc_info=True)
        return documents[:top_k] if documents else []


def get_llm_confidence_score(query: str, context: str, answer: str, llm: ChatGroq) -> Dict:
    """Güvenilirlik skoru hesaplar."""
    try:
        answer_keywords = extract_keywords(answer)
        context_keywords = extract_keywords(context)
        
        if answer_keywords:
            source_match = len(answer_keywords & context_keywords) / len(answer_keywords)
        else:
            source_match = 0.0
        
        confidence = source_match * 100
        
        if confidence >= 70:
            reasoning = "Cevap yönetmeliklere yüksek oranda dayanıyor."
        elif confidence >= 50:
            reasoning = "Cevap yönetmeliklere kısmen dayanıyor."
        else:
            reasoning = "Cevap yönetmeliklere düşük oranda dayanıyor."
        
        return {
            'confidence': confidence,
            'reasoning': reasoning,
            'source_match': source_match
        }
    
    except Exception as e:
        logger.error(f"Confidence score hatası: {str(e)}", exc_info=True)
        return {
            'confidence': 50.0,
            'reasoning': "Güven skoru hesaplanamadı.",
            'source_match': 0.5
        }


# ==================== CACHE & PERFORMANCE ====================
performance_metrics = defaultdict(list)
_retrieval_cache = {}
_cache_access_count = 0
_cache_hit_count = 0

def get_query_hash(query: str) -> str:
    return hashlib.md5(query.encode('utf-8')).hexdigest()


def keyword_search(query: str, vectorstore: FAISS, k: int = 10) -> List[Document]:
    """Geliştirilmiş keyword search - Jaccard similarity ve exact match bonus"""
    try:
        query_words = set(query.lower().split())
        query_original = query  # Orijinal query'yi sakla (büyük/küçük harf için)
        
        # İsim araması kontrolü - büyük harfli kelimeler varsa
        is_name_query = any(word[0].isupper() and len(word) > 2 for word in query.split() if word)
        
        # Daha fazla aday doküman getir
        all_docs = vectorstore.similarity_search(query, k=k*5)
        
        scored_docs = []
        for doc in all_docs:
            doc_content = doc.page_content  # Orijinal içerik (case-sensitive)
            doc_content_lower = doc_content.lower()
            doc_words = set(doc_content_lower.split())
            
            # İsim aramaları için exact match'e çok yüksek öncelik ver
            if is_name_query:
                # Orijinal query'nin exact match'i (case-sensitive)
                exact_match_bonus = 0.8 if query_original in doc_content else 0
                # Case-insensitive exact match
                exact_match_bonus += 0.5 if query.lower() in doc_content_lower else 0
                # Her kelime için exact match
                query_words_original = query_original.split()
                word_exact_match = sum(1 for word in query_words_original if word in doc_content) / len(query_words_original) * 0.3 if query_words_original else 0
                
                total_score = exact_match_bonus + word_exact_match
            else:
                # Normal arama için Jaccard similarity
                intersection = len(query_words & doc_words)
                union = len(query_words | doc_words)
                jaccard_score = intersection / union if union > 0 else 0
                
                # Exact phrase match bonus
                exact_match_bonus = 0
                if query.lower() in doc_content_lower:
                    exact_match_bonus = 0.3
                
                # Query word frequency bonus
                word_freq_bonus = sum(1 for word in query_words if word in doc_content_lower) / len(query_words) * 0.2 if query_words else 0
                
                total_score = jaccard_score + exact_match_bonus + word_freq_bonus
            
            scored_docs.append((total_score, doc))
        
        # Score'a göre sırala
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]
    except Exception as e:
        logger.error(f"Keyword search hatası: {str(e)}", exc_info=True)
        return vectorstore.similarity_search(query, k=k)


def hybrid_search(query: str, vectorstore: FAISS, semantic_k: int = None, keyword_k: int = None, final_k: int = None) -> List[Document]:
    try:
        semantic_k = semantic_k or HYBRID_SEMANTIC_K
        keyword_k = keyword_k or HYBRID_KEYWORD_K
        final_k = final_k or HYBRID_FINAL_K
        
        semantic_results = vectorstore.similarity_search(query, k=semantic_k)
        keyword_results = keyword_search(query, vectorstore, k=keyword_k)
        
        seen_ids = set()
        combined_results = []
        
        for doc in semantic_results:
            doc_id = id(doc.page_content)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_results.append(doc)
        
        for doc in keyword_results:
            doc_id = id(doc.page_content)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_results.append(doc)
        
        return combined_results[:final_k]
    except Exception as e:
        logger.error(f"Hybrid search hatası: {str(e)}", exc_info=True)
        return vectorstore.similarity_search(query, k=final_k or HYBRID_FINAL_K)


def cached_retrieval(query: str, vectorstore: FAISS, use_hybrid: bool = False) -> List[Document]:
    global _retrieval_cache, _cache_access_count, _cache_hit_count
    
    if not USE_RETRIEVAL_CACHE:
        if use_hybrid:
            return hybrid_search(query, vectorstore)
        else:
            if RETRIEVER_SEARCH_TYPE == "mmr":
                return vectorstore.max_marginal_relevance_search(
                    query, k=RETRIEVER_K, fetch_k=RETRIEVER_FETCH_K, lambda_mult=RETRIEVER_LAMBDA_MULT
                )
            else:
                return vectorstore.similarity_search(query, k=RETRIEVER_K)
    
    query_hash = get_query_hash(query)
    cache_key = f"{query_hash}_{use_hybrid}_{RETRIEVER_K}"
    
    _cache_access_count += 1
    
    if cache_key in _retrieval_cache:
        _cache_hit_count += 1
        logger.info(f"Cache hit!")
        return _retrieval_cache[cache_key]
    
    logger.info(f"Cache miss")
    
    if use_hybrid:
        results = hybrid_search(query, vectorstore)
    else:
        if RETRIEVER_SEARCH_TYPE == "mmr":
            results = vectorstore.max_marginal_relevance_search(
                query, k=RETRIEVER_K, fetch_k=RETRIEVER_FETCH_K, lambda_mult=RETRIEVER_LAMBDA_MULT
            )
        else:
            results = vectorstore.similarity_search(query, k=RETRIEVER_K)
    
    if len(_retrieval_cache) >= RETRIEVAL_CACHE_SIZE:
        oldest_key = next(iter(_retrieval_cache))
        del _retrieval_cache[oldest_key]
    
    _retrieval_cache[cache_key] = results
    
    return results


def track_performance(func_name: str, start_time: float, end_time: float, 
                     additional_metrics: Dict = None) -> Dict:
    if not TRACK_PERFORMANCE:
        return {}
    
    response_time = end_time - start_time
    
    metrics = {
        'function': func_name,
        'response_time': response_time,
        'timestamp': datetime.now().isoformat(),
        **(additional_metrics or {})
    }
    
    performance_metrics[func_name].append(metrics)
    
    return metrics


def get_performance_stats() -> Dict:
    global _cache_access_count, _cache_hit_count
    
    stats = {
        'cache_hit_rate': (_cache_hit_count / _cache_access_count * 100) if _cache_access_count > 0 else 0,
        'cache_access_count': _cache_access_count,
        'cache_hit_count': _cache_hit_count,
        'function_stats': {}
    }
    
    for func_name, metrics_list in performance_metrics.items():
        if metrics_list:
            response_times = [m['response_time'] for m in metrics_list]
            stats['function_stats'][func_name] = {
                'count': len(metrics_list),
                'avg_response_time': sum(response_times) / len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times)
            }
    
    return stats


# ==================== MAIN APP ====================
def main():
    st.set_page_config(
        page_title="Üniversite Mevzuat Asistanı", 
        page_icon="🎓", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Header
    st.markdown("<h1>🎓 Üniversite Mevzuat Asistanı</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle">
    📚 Atatürk Üniversitesi yönetmelik ve yönergeleriniz hakkında sorularınızı sorun
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Ayarlar ve Bilgiler")
        
        # Cache temizleme
        if st.button("🔄 Cache'i Temizle", use_container_width=True):
            st.cache_resource.clear()
            st.success("✅ Cache temizlendi!")
            st.rerun()
        
        st.markdown("---")
        
        # Örnek sorular
        with st.expander("💡 Örnek Sorular", expanded=False):
            st.markdown("""
            **Öğrenci İşleri:**
            - Kayıt yenileme nasıl yapılır?
            - Devamsızlık limiti nedir?
            - Mazeret sınavı için ne yapmalıyım?
            
            **Akademik:**
            - Staj süresi ne kadar?
            - Mezuniyet için kaç kredi gerekli?
            - Ders tekrarı kuralları neler?
            
            **Genel:**
            - Disiplin cezaları nelerdir?
            - Burs başvurusu nasıl yapılır?
            - Yatay geçiş şartları neler?
            """)
        
        # Performans metrikleri
        if TRACK_PERFORMANCE:
            with st.expander("📊 Performans Metrikleri", expanded=False):
                stats = get_performance_stats()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Cache Hit", f"{stats['cache_hit_rate']:.1f}%")
                with col2:
                    st.metric("Toplam Sorgu", stats['cache_access_count'])
                
                if stats['function_stats']:
                    st.markdown("#### ⏱️ Ortalama Süre")
                    for func_name, func_stats in stats['function_stats'].items():
                        st.text(f"{func_name}: {func_stats['avg_response_time']:.2f}s")
        
        st.markdown("---")
        st.markdown("### ℹ️ Hakkında")
        st.info("""
        Bu sistem yönetmelikleri analiz ederek 
        sorularınıza yanıt verir. 
        
        **Özellikler:**
        - 🔍 Akıllı arama
        - 📖 Kaynak gösterimi
        - ⚡ Hızlı yanıt
        - 🎯 Yüksek doğruluk
        """)
    
    # Session state başlat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'rag_metrics' not in st.session_state:
        st.session_state.rag_metrics = []

    # Mesaj geçmişini göster
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input
    prompt = st.chat_input("💬 Sorunuzu buraya yazın...")

    if prompt:
        # Soru doğrulama
        is_valid, error_message = validate_query(prompt)
        if not is_valid:
            st.error(f"❌ {error_message}")
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            st.session_state.messages.append({'role': 'assistant', 'content': f"❌ {error_message}"})
            st.rerun()
            return
        
        # Kullanıcı mesajını göster
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        # İsim sorusu kontrolü
        is_name_question = any(word[0].isupper() and len(word) > 2 for word in prompt.split() if word) or 'kim' in prompt.lower()
        
        # Geliştirilmiş Prompt Template
        if is_name_question:
            CUSTOM_PROMPT_TEMPLATE = """
Sen Atatürk Üniversitesi yönetmelikleri, yönergeleri ve mevzuatı konusunda uzmanlaşmış bir yapay zeka asistanısın.
Kullanıcı bir kişi hakkında soru soruyor. SADECE verilen bağlamdaki bilgilere dayanarak cevapla.

ÖNEMLİ KURALLAR (İSİM SORULARI İÇİN):
1. SADECE bağlamda verilen bilgileri kullan - ASLA bilgi uydurma
2. İsim bağlamda geçiyorsa, o kişinin unvanını, görevini ve bağlamdaki rolünü belirt
3. İsim bağlamda geçmiyorsa: "Üzgünüm, mevcut yönetmeliklerde bu kişiyle ilgili bilgi bulamadım." de
4. İlgili yönetmelik/yönerge adını belirt (PDF dosya adından)
5. İsimin geçtiği sayfa numarasını belirt
6. Eğer isim bir imza, onay veya yönetim kurulu üyesi olarak geçiyorsa bunu belirt

Bağlam:
{context}

Soru:
{question}

Cevap (SADECE bağlamdaki bilgilere dayanarak):
"""
        else:
            CUSTOM_PROMPT_TEMPLATE = """
Sen Atatürk Üniversitesi yönetmelikleri, yönergeleri ve mevzuatı konusunda uzmanlaşmış bir yapay zeka asistanısın.
Kullanıcının sorduğu soruyu cevapla. ÖNCELİKLE bağlamdaki bilgileri kullan, eğer bağlamda yeterli bilgi yoksa genel bilgilerle destekle.

⚠️ KRİTİK ADIMLAR - MUTLAKA UYGULA:
1. ADIM: Soruyu oku ve sorunun ne sorduğunu anla. Soru "başvurabilmesi için" mi diyor yoksa "devam edebilmesi için" mi diyor? Bu çok önemli!
2. ADIM: Bağlamda soruyla TAM EŞLEŞEN kelimeleri ara. Soru "başvurabilmesi için" diyorsa, bağlamda "başvurabilmesi için" kelimelerinin geçtiği cümleyi bul.
3. ADIM: Bulduğun cümledeki sayısal değeri kullan. Başka cümlelerdeki sayısal değerleri ASLA kullanma.

ÖRNEK SENARYO:
Soru: "Öğrencinin yandal programına başvurabilmesi için AGNO kaç olmalı?"
Bağlam: "...başvurabilmesi için AGNO'sunun en az 2.50 olması gerekir... devam edebilmesi için AGNO'sunun en az 2.40 olması şarttır..."
Doğru Cevap: 2.50 (çünkü soru "başvurabilmesi için" diyor ve bağlamda "başvurabilmesi için" cümlesinde 2.50 geçiyor)
Yanlış Cevap: 2.40 (çünkü bu "devam edebilmesi için" cümlesindeki değer, soru "başvurabilmesi için" soruyor)

ÖNCE SORUYU ANALİZ ET: Soruyu dikkatlice oku ve sorunun ne sorduğunu anla. Özellikle "başvurabilmesi için", "devam edebilmesi için", "başvuru şartı", "devam şartı" gibi ifadeleri tespit et. Sonra bağlamda bu ifadelerin TAM EŞLEŞMESİNİ ara.

ÖNEMLİ KURALLAR:
1. ÖNCELİK BAĞLAMDA: Bağlamda soruyla ilgili bilgi varsa, MUTLAKA bağlamdaki bilgileri kullan ve öncelik ver
2. BAĞLAMDA BİLGİ VARSA:
   - Bağlamdaki bilgileri kullanarak detaylı cevap ver
   - Madde numaralarını MUTLAKA belirt (örn: "Madde 12", "Madde 28, f.1")
   - İlgili yönetmelik/yönerge adını belirt (PDF dosya adından)
   - Bağlamdaki bilgileri kopyala-yapıştır yapma, anla ve kendi cümlelerinle açıkla
   - SAYISAL DEĞERLER İÇİN ÖZEL DİKKAT: AGNO, not ortalaması, kredi, AKTS gibi sayısal değerleri bağlamdan TAM OLARAK oku ve doğru şekilde aktar. Özellikle yandal programı için AGNO değeri bağlamda "2.50" veya "2,50" olarak geçiyorsa, bunu "2.50" olarak belirt. Bağlamda "2.40" veya "2,40" geçiyorsa "2.40" olarak belirt. Bağlamdaki değeri olduğu gibi kullan, değiştirme.
   - BAŞVURU VE DEVAM ŞARTLARI AYRIMI (ÇOK ÖNEMLİ - MUTLAKA UYULMALI): 
     * Soru "başvurabilmesi için" veya "başvuru şartı" veya "başvurabilmek için" içeriyorsa: 
       → Bağlamda "başvurabilmesi için" veya "başvurabilmek için" kelimelerinin geçtiği cümleyi bul
       → O cümledeki AGNO değerini kullan (genellikle 2.50)
       → Bağlamda "(3)" veya "üçüncü fıkra" veya "fıkra (3)" geçiyorsa ve "başvurabilmesi için" ile birlikte geçiyorsa, o fıkradaki değeri kullan
       → "Devam edebilmesi için" veya "devam" veya "(5)" veya "beşinci fıkra" veya "fıkra (5)" ile ilgili şartları ASLA kullanma
     * Soru "devam edebilmesi için" veya "devam etme şartı" veya "devam edebilmek için" içeriyorsa:
       → Bağlamda "devam edebilmesi için" veya "devam edebilmek için" kelimelerinin geçtiği cümleyi bul
       → O cümledeki AGNO değerini kullan (genellikle 2.40)
       → Bağlamda "(5)" veya "beşinci fıkra" veya "fıkra (5)" geçiyorsa ve "devam edebilmesi için" ile birlikte geçiyorsa, o fıkradaki değeri kullan
       → "Başvurabilmesi için" veya "(3)" veya "üçüncü fıkra" veya "fıkra (3)" ile ilgili şartları ASLA kullanma
     * KRİTİK: Soru kelimelerini bağlamdaki cümlelerle TAM EŞLEŞTİR. "başvurabilmesi için" sorusu sorulduğunda, bağlamda "başvurabilmesi için" kelimelerinin geçtiği cümleyi bul ve o cümledeki sayısal değeri kullan. Başka bir cümledeki (örneğin "devam edebilmesi için" cümlesindeki) sayısal değeri ASLA kullanma.
     * ÖRNEK: Soru "yandal programına başvurabilmesi için AGNO kaç olmalı?" ise ve bağlamda hem "(3) başvurabilmesi için AGNO'sunun en az 2.50 olması" hem de "(5) devam edebilmesi için AGNO'sunun en az 2.40 olması" geçiyorsa, SADECE "(3)" fıkrasındaki "başvurabilmesi için" cümlesindeki 2.50 değerini kullan, "(5)" fıkrasındaki 2.40 değerini kullanma.
3. BAĞLAMDA BİLGİ YOKSA VEYA YETERSİZSE:
   - Önce şunu belirt: "Mevcut yönetmeliklerde bu konuyla ilgili spesifik bilgi bulunmamaktadır."
   - Sonra genel bilgi verebilirsin ama bunu açıkça belirt: "Genel olarak bilindiği üzere..." veya "Üniversite yönetmeliklerinde genellikle..."
   - Genel bilgi verirken spekülatif olma, sadece yaygın bilinen bilgileri paylaş
4. Cevabı yapılandırılmış ve net ver:
   - Önce kısa özet
   - Sonra detaylı açıklama
   - Gerekirse madde numaraları ve şartlar
5. Bağlamda birden fazla yönetmelik varsa, en ilgili olanına öncelik ver
6. Açık ve Uzaktan Öğretim Fakültesi yönetmelikleri sadece açık öğretim öğrencileri için geçerlidir
7. Genel öğrenciler için "Atatürk Üniversitesi Önlisans ve Lisans Eğitim Öğretim ve Sınav Yönetmeliği" önceliklidir
8. BAĞLAM KONTROLÜ: Bağlamı dikkatlice oku. Eğer bağlamda soruyla ilgili bilgi varsa (eş anlamlılar dahil), o bilgileri MUTLAKA kullan

Bağlam:
{context}

Soru:
{question}

Cevap (Önce bağlamdaki bilgileri kullan, yoksa genel bilgiyle destekle):

⚠️ CEVAP VERMEDEN ÖNCE KONTROL ET:
1. Soru "başvurabilmesi için" mi diyor? 
   → Bağlamda "başvurabilmesi için" kelimelerinin geçtiği cümleyi bul
   → O cümledeki sayısal değeri kullan (genellikle 2.50)
   → Bağlamda "(3)" veya "üçüncü fıkra" veya "fıkra (3)" ve "başvurabilmesi için" birlikte geçiyorsa, o fıkradaki değeri kullan
   → "(5)" veya "beşinci fıkra" veya "devam edebilmesi için" ile ilgili değerleri ASLA kullanma
2. Soru "devam edebilmesi için" mi diyor?
   → Bağlamda "devam edebilmesi için" kelimelerinin geçtiği cümleyi bul
   → O cümledeki sayısal değeri kullan (genellikle 2.40)
   → Bağlamda "(5)" veya "beşinci fıkra" veya "fıkra (5)" ve "devam edebilmesi için" birlikte geçiyorsa, o fıkradaki değeri kullan
   → "(3)" veya "üçüncü fıkra" veya "başvurabilmesi için" ile ilgili değerleri ASLA kullanma
3. Eğer soru "başvurabilmesi için" diyorsa ve bağlamda hem "(3) başvurabilmesi için 2.50" hem de "(5) devam edebilmesi için 2.40" geçiyorsa → SADECE (3) fıkrasındaki 2.50 kullan, (5) fıkrasındaki 2.40 kullanma!
4. Eğer soru "devam edebilmesi için" diyorsa ve bağlamda hem "(3) başvurabilmesi için 2.50" hem de "(5) devam edebilmesi için 2.40" geçiyorsa → SADECE (5) fıkrasındaki 2.40 kullan, (3) fıkrasındaki 2.50 kullanma!

ÖNEMLİ: Soruyu tekrar oku. Soru "başvurabilmesi için" diyorsa, bağlamda "başvurabilmesi için" kelimelerinin geçtiği cümleyi bul ve o cümledeki sayısal değeri kullan. Soru "devam edebilmesi için" diyorsa, bağlamda "devam edebilmesi için" kelimelerinin geçtiği cümleyi bul ve o cümledeki sayısal değeri kullan. Başka bir cümledeki sayısal değeri ASLA kullanma.
"""
        
        try:
            with st.chat_message('assistant'):
                with st.spinner("🔍 Yönetmelikler taranıyor..."):
                    vectorstore = get_vectorstore()
                    if vectorstore is None:
                        st.error("❌ Vector store yüklenemedi")
                        return

                    llm_rag = get_llm()
                    
                    # Geliştirilmiş Query Expansion - Üniversite yönetmelikleri için özelleştirilmiş
                    expanded_query = prompt.lower()
                    expansion_terms = []
                    
                    # İsim araması kontrolü - "kim" sorusu veya büyük harfli kelimeler (isim olabilir)
                    is_name_query = False
                    query_words = prompt.split()
                    
                    # Büyük harfle başlayan kelimeler varsa (isim olabilir)
                    has_capitalized_words = any(word[0].isupper() and len(word) > 2 for word in query_words if word)
                    has_kim = 'kim' in expanded_query or 'kimdir' in expanded_query
                    
                    if has_kim or has_capitalized_words:
                        is_name_query = True
                        # İsim aramaları için exact match öncelikli - expansion yapma
                        final_query = prompt  # Orijinal query'yi kullan, expansion yapma
                        logger.info(f"İsim araması tespit edildi: {prompt}")
                    else:
                        # Terim eşleştirmeleri
                        term_expansions = {
                            'staj': ['staj', 'uygulama', 'mesleki eğitim', 'pratik eğitim'],
                            'sınav': ['sınav', 'muayene', 'değerlendirme', 'imtihan'],
                            'devamsızlık': ['devamsızlık', 'yoklama', 'katılım', 'derse devam'],
                            'kayıt': ['kayıt', 'tescil', 'ders kaydı', 'kayıt yenileme', 'kayıt dondurma'],
                            'mezuniyet': ['mezuniyet', 'diploma', 'bitirme', 'mezun'],
                            'mazeret': ['mazeret', 'özür', 'geçerli sebep', 'haklı neden'],
                            'yaz okulu': ['yaz okulu', 'yaz dönemi', 'yaz öğretimi'],
                            'yatay geçiş': ['yatay geçiş', 'kurumlar arası geçiş', 'programlar arası geçiş'],
                            'çift anadal': ['çift anadal', 'çap', 'yan dal', 'yandal'],
                            'disiplin': ['disiplin', 'ceza', 'yaptırım', 'uyarı'],
                            'kredi': ['kredi', 'akts', 'ders kredisi', 'kredi transferi'],
                            'ders': ['ders', 'müfredat', 'program', 'eğitim'],
                            'öğrenci': ['öğrenci', 'talebe', 'öğrenim gören'],
                            'fakülte': ['fakülte', 'bölüm', 'program', 'anabilim dalı'],
                        }
                        
                        # Query'deki terimleri genişlet
                        for term, synonyms in term_expansions.items():
                            if term in expanded_query:
                                expansion_terms.extend(synonyms)
                        
                        # Orijinal query'yi de ekle
                        expansion_terms.append(prompt)
                        
                        # Benzersiz terimleri birleştir
                        if expansion_terms:
                            final_query = ' '.join(set(expansion_terms))
                        else:
                            final_query = prompt
                    
                    # Retrieval
                    retrieval_start_time = time.time()
                    
                    # İsim aramaları için hybrid search'i bypass et - direkt exact match
                    if is_name_query:
                        use_hybrid = False  # İsim aramaları için hybrid search kullanma
                        # İsim aramaları için keyword search ile exact match'e öncelik ver
                        logger.info(f"İsim araması için exact match modu aktif: {prompt}")
                        source_documents = keyword_search(final_query, vectorstore, k=RETRIEVER_K * 2)  # 2x daha fazla doküman, exact match öncelikli
                        
                        # Retriever oluştur
                        from langchain_core.retrievers import BaseRetriever
                        from langchain_core.callbacks import CallbackManagerForRetrieverRun
                        from pydantic import Field
                        
                        class ExactMatchRetriever(BaseRetriever):
                            docs: List[Document] = Field(default_factory=list)
                            
                            def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None):
                                return self.docs
                        
                        retriever = ExactMatchRetriever(docs=source_documents)
                    else:
                        use_hybrid = USE_HYBRID_SEARCH
                        if use_hybrid:
                            source_documents = cached_retrieval(final_query, vectorstore, use_hybrid=True)
                            
                            from langchain_core.retrievers import BaseRetriever
                            from langchain_core.callbacks import CallbackManagerForRetrieverRun
                            from pydantic import Field
                            
                            class HybridRetriever(BaseRetriever):
                                docs: List[Document] = Field(default_factory=list)
                                
                                def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None):
                                    return self.docs
                            
                            retriever = HybridRetriever(docs=source_documents)
                        else:
                            source_documents = cached_retrieval(final_query, vectorstore, use_hybrid=False)
                            
                            if RETRIEVER_SEARCH_TYPE == "mmr":
                                retriever = vectorstore.as_retriever(
                                    search_type="mmr",
                                    search_kwargs={
                                        'k': RETRIEVER_K,
                                        'fetch_k': RETRIEVER_FETCH_K,
                                        'lambda_mult': RETRIEVER_LAMBDA_MULT
                                    }
                                )
                            else:
                                retriever = vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={'k': RETRIEVER_K}
                                )
                    
                    # Re-ranking
                    if USE_RERANKING and source_documents:
                        source_documents = rerank_documents(final_query, source_documents, top_k=RERANK_TOP_K)
                    
                    track_performance("retrieval", retrieval_start_time, time.time(), {
                        'query': final_query[:50],
                        'doc_count': len(source_documents)
                    })
                    
                    # Token limit kontrolü
                    if source_documents:
                        total_tokens = sum(estimate_tokens(doc.page_content) for doc in source_documents)
                        prompt_tokens = estimate_tokens(CUSTOM_PROMPT_TEMPLATE) + estimate_tokens(final_query)
                        total_with_prompt = total_tokens + prompt_tokens
                        
                        safe_limit = int(MAX_CONTEXT_TOKENS * 0.9)
                        
                        if total_with_prompt > safe_limit:
                            source_documents = limit_context_size(source_documents, safe_limit - prompt_tokens)
                        else:
                            if not use_hybrid:
                                if RETRIEVER_SEARCH_TYPE == "mmr":
                                    retriever = vectorstore.as_retriever(
                                        search_type="mmr",
                                        search_kwargs={
                                            'k': min(len(source_documents), RETRIEVER_K),
                                            'fetch_k': min(len(source_documents) * 2, RETRIEVER_FETCH_K),
                                            'lambda_mult': RETRIEVER_LAMBDA_MULT
                                        }
                                    )
                                else:
                                    retriever = vectorstore.as_retriever(
                                        search_type="similarity",
                                        search_kwargs={'k': min(len(source_documents), RETRIEVER_K)}
                                    )
                            else:
                                from langchain_core.retrievers import BaseRetriever
                                from langchain_core.callbacks import CallbackManagerForRetrieverRun
                                from pydantic import Field
                                
                                class HybridRetriever(BaseRetriever):
                                    docs: List[Document] = Field(default_factory=list)
                                    
                                    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None):
                                        return self.docs
                                
                                retriever = HybridRetriever(docs=source_documents)
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm_rag,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                    )
                    
                    if not source_documents or len(source_documents) == 0:
                        # Doküman bulunamadı ama LLM genel bilgi verebilir
                        logger.info("Doküman bulunamadı, LLM genel bilgi verecek")
                        # Boş bağlam ile LLM'e soru sor (genel bilgi verebilir)
                        empty_context_prompt = f"""Sen Atatürk Üniversitesi yönetmelikleri konusunda uzman bir asistanısın. 
Kullanıcının sorusunu cevapla. Mevcut yönetmeliklerde bu konuyla ilgili spesifik bilgi bulunmamaktadır, 
ancak genel bilgilerle yardımcı olabilirsin. Genel bilgi verirken bunu açıkça belirt.

Soru: {prompt}

Cevap:"""
                        try:
                            llm_response = llm_rag.invoke(empty_context_prompt)
                            if isinstance(llm_response, dict):
                                result = llm_response.get("content", str(llm_response))
                            elif hasattr(llm_response, 'content'):
                                result = llm_response.content
                            else:
                                result = str(llm_response)
                        except Exception as e:
                            logger.error(f"LLM genel bilgi verirken hata: {e}")
                            result = f"Mevcut yönetmeliklerde bu konuyla ilgili spesifik bilgi bulunmamaktadır. Lütfen daha detaylı bir soru sorun."
                        st.info("ℹ️ Bu cevap genel bilgilere dayanmaktadır. Spesifik yönetmelik bilgisi için lütfen daha detaylı sorun.")
                        source_documents = []  # Boş liste olarak ayarla
                    else:
                        # İsim aramaları için özel kontrol - isim dokümanlarda var mı?
                        if is_name_query:
                            # İsimin dokümanlarda geçip geçmediğini kontrol et
                            query_name_parts = [word for word in prompt.split() if word[0].isupper() and len(word) > 2]
                            name_found_in_docs = False
                            for doc in source_documents:
                                doc_content_lower = doc.page_content.lower()
                                for name_part in query_name_parts:
                                    if name_part.lower() in doc_content_lower:
                                        name_found_in_docs = True
                                        break
                                if name_found_in_docs:
                                    break
                            
                            if not name_found_in_docs:
                                # İsim bulunamadı - daha fazla doküman ara
                                logger.info(f"İsim dokümanlarda bulunamadı, genişletilmiş arama yapılıyor...")
                                extended_docs = vectorstore.similarity_search(final_query, k=RETRIEVER_K * 3)
                                source_documents = keyword_search(final_query, vectorstore, k=RETRIEVER_K * 2)
                        
                        try:
                            llm_start_time = time.time()
                            response = qa_chain.invoke({'query': final_query})
                            result = response["result"]
                            response_source_documents = response["source_documents"]  # Response'dan gelen dokümanlar
                            
                            # Retrieval'dan gelen dokümanları sakla (fallback için)
                            retrieval_source_documents = source_documents.copy() if source_documents else []
                            
                            track_performance("llm_generation", llm_start_time, time.time(), {
                                'query': final_query[:50],
                                'response_length': len(result)
                            })
                            
                            # İsim aramaları için özel kontrol - LLM cevap vermedi mi?
                            if is_name_query and ("bulamadım" in result.lower() or "bilgi yok" in result.lower() or "bilgi bulamadım" in result.lower()):
                                # LLM cevap vermedi, direkt dokümanlardan bilgi çıkar
                                logger.info(f"LLM cevap vermedi, dokümanlardan direkt bilgi çıkarılıyor... is_name_query={is_name_query}")
                                name_info = []
                                query_name_parts = [word for word in prompt.split() if word[0].isupper() and len(word) > 2]
                                
                                # Önce retrieval'dan gelen dokümanları kontrol et, sonra response'dan gelenleri
                                docs_to_check = retrieval_source_documents + response_source_documents
                                logger.info(f"İsim parçaları: {query_name_parts}, Retrieval doküman sayısı: {len(retrieval_source_documents)}, Response doküman sayısı: {len(response_source_documents)}, Toplam: {len(docs_to_check)}")
                                
                                for doc in docs_to_check:
                                    doc_content = doc.page_content
                                    doc_content_lower = doc_content.lower()
                                    
                                    for name_part in query_name_parts:
                                        if name_part.lower() in doc_content_lower:
                                            # F-string içinde backslash kullanılamaz
                                            source_path = doc.metadata.get('source', '') if doc.metadata.get('source') else ''
                                            source_name = source_path.split('\\')[-1] if source_path else 'Bilinmiyor'
                                            logger.info(f"İsim parçası '{name_part}' doküman içinde bulundu: {source_name}")
                                            # İsimin geçtiği satırı bul
                                            lines = doc_content.split('\n')
                                            for i, line in enumerate(lines):
                                                if name_part.lower() in line.lower():
                                                    logger.info(f"İsim satır {i}'de bulundu: {line[:100]}")
                                                    # İsimin yanındaki bilgileri al (öncesi ve sonrası)
                                                    context_lines = []
                                                    # Önceki 2 satır
                                                    for j in range(max(0, i-2), i):
                                                        if lines[j].strip():
                                                            context_lines.append(lines[j].strip())
                                                    # İsimin geçtiği satır
                                                    context_lines.append(line.strip())
                                                    # Sonraki 2 satır
                                                    for j in range(i+1, min(len(lines), i+3)):
                                                        if lines[j].strip():
                                                            context_lines.append(lines[j].strip())
                                                    
                                                    if context_lines:
                                                        name_info.append(' '.join(context_lines))
                                                        logger.info(f"İsim bilgisi eklendi: {len(context_lines)} satır")
                                                    break
                                    
                                    if name_info:
                                        break
                                
                                if name_info:
                                    result = f"Bağlamda bulunan bilgi:\n\n"
                                    for info in name_info[:2]:  # İlk 2 bilgiyi göster
                                        result += f"{info}\n\n"
                                    # F-string içinde backslash kullanılamaz
                                    first_doc = docs_to_check[0] if docs_to_check else None
                                    if first_doc:
                                        source_path = first_doc.metadata.get('source', '')
                                        source_name = source_path.split('\\')[-1] if source_path else 'Bilinmiyor'
                                        page_num = first_doc.metadata.get('page', '?')
                                    else:
                                        source_name = 'Bilinmiyor'
                                        page_num = '?'
                                    result += f"Kaynak: {source_name} (Sayfa {page_num})"
                                    # Response dokümanlarını güncelle
                                    source_documents = response_source_documents
                                else:
                                    # Fallback çalışmadı, response dokümanlarını kullan
                                    source_documents = response_source_documents
                            else:
                                # Normal durum, response dokümanlarını kullan
                                source_documents = response_source_documents
                            
                            # Halüsinasyon kontrolü
                            if result and source_documents and not result.startswith("Üzgünüm") and not result.startswith("Bağlamda"):
                                hallucination_check = check_hallucination(result, source_documents)
                                context_text = ' '.join([doc.page_content for doc in source_documents])
                                llm_confidence = get_llm_confidence_score(final_query, context_text, result, llm_rag)
                                combined_confidence = (hallucination_check['confidence'] * 0.6 + llm_confidence['confidence'] / 100 * 0.4) * 100
                                
                                if combined_confidence < 15:
                                    st.warning(f"⚠️ Düşük güvenilirlik ({combined_confidence:.1f}/100)")
                                else:
                                    rag_metric = {
                                        'query': prompt[:50],
                                        'confidence': combined_confidence,
                                        'match_ratio': hallucination_check['match_ratio'],
                                        'response_length': len(result),
                                        'doc_count': len(source_documents),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    st.session_state.rag_metrics.append(rag_metric)
                                
                                # Metrikler sidebar'da
                                if combined_confidence >= 20:
                                    with st.sidebar.expander("📊 Son Sorgu Analizi", expanded=False):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Güvenilirlik", f"{combined_confidence:.0f}/100")
                                        with col2:
                                            st.metric("Doküman", len(source_documents))
                        except Exception as e:
                            st.error(f"❌ Hata: {str(e)[:100]}")
                            result = "Üzgünüm, bir hata oluştu."
                    
                    # Kaynakları formatla ve filtrele
                    unique_sources = {}
                    prompt_lower = prompt.lower()
                    greetings = ['merhaba', 'selam', 'iyi günler']
                    is_greeting = any(greeting in prompt_lower for greeting in greetings)
                    
                    # Açık öğretim ile ilgili soruları tespit et
                    is_open_education_query = any(term in prompt_lower for term in [
                        'açık öğretim', 'uzaktan eğitim', 'açık ve uzaktan'
                    ])
                    
                    if not is_greeting and source_documents:
                        for doc in source_documents:
                            source_name = doc.metadata.get('source', '').split('\\')[-1] if doc.metadata.get('source') else 'Bilinmiyor'
                            
                            # Genel sorularda açık öğretim yönetmeliklerini filtrele
                            if not is_open_education_query:
                                if 'Açık ve Uzaktan Öğretim' in source_name or 'Açık Yükseköğretim' in source_name:
                                    continue
                            
                            page_num = doc.metadata.get('page', 'Bilinmiyor')
                            if source_name not in unique_sources:
                                unique_sources[source_name] = []
                            if page_num not in unique_sources[source_name]:
                                unique_sources[source_name].append(page_num)
                        
                        # En çok referans alan kaynakları göster
                        sorted_sources = sorted(unique_sources.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                        
                        if sorted_sources:
                            source_info = "\n\n---\n\n**📖 Kaynaklar:**\n"
                            for i, (source_name, pages) in enumerate(sorted_sources, 1):
                                # PDF adını kısalt ve daha okunabilir yap
                                display_name = source_name.replace('.pdf', '')
                                if len(display_name) > 80:
                                    display_name = display_name[:77] + '...'
                                
                                first_page = sorted(set(pages))[0] if pages else 'Bilinmiyor'
                                source_info += f"{i}. {display_name} (Sayfa {first_page})\n"
                            
                            if len(unique_sources) > 5:
                                source_info += f"\n*... ve {len(unique_sources) - 5} kaynak daha*\n"
                            
                            result_to_show = result + source_info
                        else:
                            result_to_show = result
                    else:
                        result_to_show = result
                    
                    st.markdown(result_to_show)
                    
                    if not is_greeting and source_documents:
                        st.success(f"✅ {len(source_documents)} yönetmelik parçasından bilgi çekildi ({len(unique_sources)} kaynak)")
                    else:
                        st.success("✅ Cevap oluşturuldu")
            
            st.session_state.messages.append({
                'role':'assistant', 
                'content': result_to_show
            })

        except Exception as e:
            st.error(f"❌ Hata: {str(e)}")
            import traceback
            with st.expander("🔍 Detaylı Hata"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
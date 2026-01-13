"""
Konfigürasyon dosyası - Tüm ayarlar burada toplanmıştır
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Proje root dizini (mutlak yol olarak çözümlenir)
# __file__ = src/config.py olduğu için parent.parent = proje root
_config_file = Path(__file__).resolve()
PROJECT_ROOT = _config_file.parent.parent.resolve()

# Debug: Path'lerin doğru olduğundan emin ol
# Eğer vectorstore klasörü bulunamazsa, alternatif yolları dene
if not (PROJECT_ROOT / "vectorstore").exists():
    # Alternatif: run.py veya app.py'nin bulunduğu dizini kullan
    _current_file = Path(__file__).resolve()
    _possible_roots = [
        _current_file.parent.parent,  # src/config.py -> src -> root
        Path.cwd(),  # Current working directory
        Path(__file__).parent.parent.parent,  # Bir üst dizin daha (eğer farklı yapı varsa)
    ]
    for root in _possible_roots:
        root_resolved = root.resolve()
        if (root_resolved / "vectorstore").exists():
            PROJECT_ROOT = root_resolved
            logger.info(f"PROJECT_ROOT otomatik olarak bulundu: {PROJECT_ROOT}")
            break

# Vector Store Ayarları (proje root'una göre)
DB_FAISS_PATH = str(PROJECT_ROOT / "vectorstore" / "db_faiss")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Doküman İşleme Ayarları
CHUNK_SIZE = 200  # Küçültüldü - daha iyi precision için (isimler, küçük bilgiler için)
CHUNK_OVERLAP = 100  # Overlap da küçültüldü (chunk_size'ın %20'si)  

# Retrieval Ayarları
RETRIEVER_K = 15  # En benzer kaç doküman parçası getirilecek (MMR için) - Artırıldı
RETRIEVER_FETCH_K = 30  # MMR için aday doküman sayısı - Artırıldı
RETRIEVER_LAMBDA_MULT = 0.5  # MMR için relevance vs diversity dengesi (0.0-1.0) - Daha fazla çeşitlilik
RETRIEVER_SEARCH_TYPE = "mmr"  

# Token Limit Ayarları
MAX_CONTEXT_TOKENS = 4000  
TOKEN_ESTIMATION_FACTOR = 1.3 

# LLM Ayarları
LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
TEMPERATURE_RAG = 0.2  
TEMPERATURE_NO_RAG = 0.7  

# UI Ayarları
MAX_QUERY_LENGTH = 1000  
MIN_QUERY_LENGTH = 3  

# Data Path (proje root'una göre)
DATA_PATH = str(PROJECT_ROOT / "data")

# Re-ranking Ayarları
USE_RERANKING = True  
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  
RERANK_TOP_K = 15  # Artırıldı - daha fazla doküman re-rank edilsin  

# Hybrid Search Ayarları
USE_HYBRID_SEARCH = True  
HYBRID_SEMANTIC_K = 20  # Artırıldı - daha fazla semantic sonuç
HYBRID_KEYWORD_K = 20  # Artırıldı - daha fazla keyword sonuç
HYBRID_FINAL_K = 15  # Artırıldı - daha fazla final sonuç  

# Retrieval Cache Ayarları
USE_RETRIEVAL_CACHE = True 
RETRIEVAL_CACHE_SIZE = 100  

# Performance Tracking Ayarları
TRACK_PERFORMANCE = True  
PERFORMANCE_LOG_FILE = str(PROJECT_ROOT / "logs" / "performance_logs.json")  # Performans log dosyası

# Citation Ayarları
USE_INLINE_CITATIONS = False 
CITATION_THRESHOLD = 0.5  
MAX_CITATIONS_PER_SENTENCE = 2  


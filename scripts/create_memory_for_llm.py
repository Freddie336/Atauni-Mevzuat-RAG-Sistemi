from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

# Config dosyasını import et
import sys
from pathlib import Path
# Proje root'unu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DB_FAISS_PATH
except ImportError:
    # Fallback değerler (mutlak yol ile)
    project_root = Path(__file__).parent.parent.resolve()
    DATA_PATH = str(project_root / "data")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DB_FAISS_PATH = str(project_root / "vectorstore" / "db_faiss")

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Step 1: Load raw PDF(s) and TXT files
def load_all_files(data):
    from pathlib import Path
    import os
    
    documents = []
    pdf_files = list(Path(data).glob("*.pdf"))
    txt_files = list(Path(data).glob("*.txt"))
    
    logger.info(f"Toplam {len(pdf_files)} PDF dosyasi ve {len(txt_files)} TXT dosyasi bulundu.")
    logger.info("Dosyalar yukleniyor...")
    
    # PDF dosyalarını yükle
    for pdf_file in pdf_files:
        try:
            logger.info(f"  Yukleniyor: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"    [OK] {len(docs)} sayfa yuklendi")
        except Exception as e:
            logger.error(f"    [HATA] {pdf_file.name} yuklenemedi: {str(e)}", exc_info=True)
            logger.warning(f"    Bu dosya atlaniyor...")
            continue
    
    # TXT dosyalarını yükle
    for txt_file in txt_files:
        try:
            logger.info(f"  Yukleniyor: {txt_file.name}")
            loader = TextLoader(str(txt_file), encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"    [OK] {len(docs)} dokuman yuklendi")
        except Exception as e:
            logger.error(f"    [HATA] {txt_file.name} yuklenemedi: {str(e)}", exc_info=True)
            logger.warning(f"    Bu dosya atlaniyor...")
            continue
    
    return documents

documents = load_all_files(data=DATA_PATH)
logger.info(f"\nToplam {len(documents)} dokuman yuklendi.")


# Step 2: Create Chunks
def create_chunks(extracted_data):
    logger.info(f"Chunk'lar oluşturuluyor: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    logger.info(f"Toplam {len(text_chunks)} chunk oluşturuldu.")
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)

# Step 3: Create Vector Embeddings 

def get_embedding_model():
    logger.info(f"Embedding modeli yükleniyor: {EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info("Embedding modeli başarıyla yüklendi.")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
logger.info(f"FAISS vector store oluşturuluyor: {DB_FAISS_PATH}")
try:
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    logger.info(f"Vector store başarıyla kaydedildi: {DB_FAISS_PATH}")
    print(f"\n✅ Başarılı! Vector store oluşturuldu: {DB_FAISS_PATH}")
    print(f"   Toplam {len(text_chunks)} chunk embed edildi ve kaydedildi.")
except Exception as e:
    logger.error(f"Vector store kaydedilemedi: {str(e)}", exc_info=True)
    raise
"""İsim araması testi"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local('vectorstore/db_faiss', embeddings, allow_dangerous_deserialization=True)

query = "Batıkan AKSOY"
print(f"Arama: {query}\n")

results = db.similarity_search(query, k=10)
print(f"Bulunan {len(results)} sonuc:\n")

for i, doc in enumerate(results, 1):
    source = doc.metadata.get('source', '').split('\\')[-1] if doc.metadata.get('source') else 'Bilinmiyor'
    page = doc.metadata.get('page', '?')
    content_lower = doc.page_content.lower()
    
    # İsim var mı kontrol et
    has_name = 'batıkan' in content_lower or 'aksoy' in content_lower
    marker = "[VAR]" if has_name else "[YOK]"
    
    print(f"{i}. {source} - Sayfa {page} {marker}")
    if has_name:
        # İsimin geçtiği kısmı bul
        lines = doc.page_content.split('\n')
        for line in lines:
            if 'batıkan' in line.lower() or 'aksoy' in line.lower():
                print(f"   -> {line[:200]}")
                break
    print()


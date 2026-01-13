"""Atatürk Üniversitesi kuruluş tarihi kontrolü"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Vector store'u yükle
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local('vectorstore/db_faiss', embeddings, allow_dangerous_deserialization=True)

# Farklı sorgular dene
queries = [
    "Atatürk Üniversitesi kuruluş tarihi",
    "Atatürk Üniversitesi ne zaman kuruldu",
    "1957",
    "kuruluş",
    "Yükseköğretim Kanunu 2547"
]

print("=" * 80)
print("KURULUS TARIHI ARAMA TESTI")
print("=" * 80)

for query in queries:
    print(f"\n{'='*80}")
    print(f"Sorgu: {query}")
    print(f"{'='*80}")
    
    results = db.similarity_search(query, k=5)
    print(f"Bulunan {len(results)} sonuc:\n")
    
    found_relevant = False
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', '').split('\\')[-1] if doc.metadata.get('source') else 'Bilinmiyor'
        page = doc.metadata.get('page', '?')
        content_preview = doc.page_content[:200].replace('\n', ' ')
        
        # İlgili içerik var mı kontrol et
        content_lower = doc.page_content.lower()
        has_kurulus = any(term in content_lower for term in ['kuruluş', '1957', 'kuruldu', 'tarih'])
        
        marker = "[ILGILI]" if has_kurulus else "[ILGISIZ]"
        print(f"{i}. {source} - Sayfa {page} {marker}")
        if has_kurulus:
            found_relevant = True
            print(f"   İçerik: {content_preview}...")
        print()
    
    if not found_relevant:
        print("   -> Bu sorgu için ilgili içerik bulunamadı")

print("\n" + "=" * 80)
print("SONUC:")
print("=" * 80)
print("Eğer hiçbir sorguda 'kuruluş' veya '1957' gibi terimler bulunamadıysa,")
print("sistemin 'bilgi bulamadım' cevabı DOĞRUDUR.")
print("Yönetmelikler genellikle kuruluş tarihi gibi tarihsel bilgileri içermez.")


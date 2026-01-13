"""
Zor Test Soruları - PDF'lerden çıkarılan spesifik bilgiler
Bu soruları doğru cevaplarsa uygulama başarılıdır.
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

# Vector store'u yükle
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local('vectorstore/db_faiss', embeddings, allow_dangerous_deserialization=True)

# Zor test soruları - PDF'lerden çıkarılan spesifik bilgiler
HARD_QUESTIONS = [
    # Spesifik tarih ve sayı soruları
    "2547 sayılı kanunun 44. maddesi nedir?",
    "Öğrenci disiplin işlemleri hangi kanun maddesine göre yürütülür?",
    "Staj süresi minimum kaç iş günüdür?",
    "Yaz okulu için başvuru tarihleri ne zaman?",
    
    # Spesifik kurallar ve limitler
    "Devamsızlık limiti yüzde kaçtır?",
    "Mezuniyet için minimum kaç kredi gerekir?",
    "Ders tekrarı için maksimum kaç kez sınav hakkı var?",
    "Yatay geçiş için minimum not ortalaması kaçtır?",
    
    # Spesifik prosedürler
    "Mazeret sınavı için başvuru süresi kaç gündür?",
    "Kayıt dondurma için başvuru nereye yapılır?",
    "Çift anadal başvurusu için hangi dönemde yapılır?",
    "Diploma almak için hangi belgeler gereklidir?",
    
    # Spesifik unvanlar ve görevler
    "Genel Sekreter kimdir?",
    "Başkan vekili kimdir?",
    "Rektör yardımcısı kimdir?",
    
    # Spesifik yönetmelik maddeleri
    "Öğrenci disiplin yönetmeliğinin 5. maddesi nedir?",
    "Eğitim öğretim yönetmeliğinin 12. maddesi nedir?",
    "Staj yönetmeliğinin 3. maddesi nedir?",
    
    # Spesifik şartlar ve koşullar
    "Yatay geçiş için hangi şartlar gereklidir?",
    "Çift anadal için minimum not ortalaması kaçtır?",
    "Yaz okuluna kimler başvurabilir?",
    "Kayıt dondurma için hangi şartlar gereklidir?",
    
    # Spesifik süreler ve limitler
    "Azami öğrenim süresi kaç yıldır?",
    "Staj süresi maksimum kaç gündür?",
    "Ders kayıt süresi kaç gündür?",
    "Mazeret sınavı için başvuru süresi ne kadar?",
]

def test_question(query, expected_keywords=None):
    """Bir soruyu test et"""
    print(f"\n{'='*80}")
    print(f"SORU: {query}")
    print(f"{'='*80}")
    
    results = db.similarity_search(query, k=5)
    print(f"Bulunan {len(results)} sonuç:\n")
    
    found_relevant = False
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', '').split('\\')[-1] if doc.metadata.get('source') else 'Bilinmiyor'
        page = doc.metadata.get('page', '?')
        content_preview = doc.page_content[:200].replace('\n', ' ')
        
        # Beklenen kelimeler var mı kontrol et
        if expected_keywords:
            content_lower = doc.page_content.lower()
            has_keywords = any(keyword.lower() in content_lower for keyword in expected_keywords)
            marker = "[ILGILI]" if has_keywords else "[ILGISIZ]"
            if has_keywords:
                found_relevant = True
        else:
            marker = ""
        
        print(f"{i}. {source} - Sayfa {page} {marker}")
        print(f"   İçerik: {content_preview}...")
        print()
    
    if expected_keywords and not found_relevant:
        print(f"   ⚠️ Beklenen kelimeler bulunamadı: {expected_keywords}")
    
    return found_relevant

if __name__ == "__main__":
    print("=" * 80)
    print("ZOR TEST SORULARI - PDF'LERDEN ÇIKARILAN SPESİFİK BİLGİLER")
    print("=" * 80)
    print(f"\nToplam {len(HARD_QUESTIONS)} zor soru hazırlandı.\n")
    
    # Bazı soruları test et
    test_cases = [
        ("Staj süresi minimum kaç iş günüdür?", ["20", "iş günü", "minimum"]),
        ("2547 sayılı kanunun 44. maddesi nedir?", ["44", "madde", "2547"]),
        ("Devamsızlık limiti yüzde kaçtır?", ["devamsızlık", "limit", "yüzde"]),
        ("Genel Sekreter kimdir?", ["Genel Sekreter", "Başkan"]),
    ]
    
    print("\n" + "=" * 80)
    print("TEST SONUÇLARI:")
    print("=" * 80)
    
    for query, keywords in test_cases:
        found = test_question(query, keywords)
        if found:
            print("✅ İlgili içerik bulundu")
        else:
            print("❌ İlgili içerik bulunamadı")
    
    print("\n" + "=" * 80)
    print("TÜM ZOR SORULAR:")
    print("=" * 80)
    for i, q in enumerate(HARD_QUESTIONS, 1):
        print(f"{i:2d}. {q}")


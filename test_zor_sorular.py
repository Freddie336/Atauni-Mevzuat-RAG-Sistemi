"""
Zor Test Soruları - PDF'lerden çıkarılan spesifik bilgiler
Bu soruları doğru cevaplarsa uygulama başarılıdır.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Vector store'u yükle
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local('vectorstore/db_faiss', embeddings, allow_dangerous_deserialization=True)

# EN ZOR TEST SORULARI - Kritik Test
CRITICAL_QUESTIONS = [
    {
        "soru": "Batıkan AKSOY kimdir?",
        "beklenen_kelimeler": ["Batıkan", "AKSOY", "Genel Sekreter", "Başkan"],
        "tip": "isim_bulma",
        "zorluk": "ÇOK ZOR"
    },
    {
        "soru": "2547 sayılı kanunun 44. maddesi nedir?",
        "beklenen_kelimeler": ["44", "madde", "2547"],
        "tip": "spesifik_madde",
        "zorluk": "ÇOK ZOR"
    },
    {
        "soru": "Staj süresi minimum kaç iş günüdür?",
        "beklenen_kelimeler": ["20", "iş günü", "minimum", "staj"],
        "tip": "spesifik_sayi",
        "zorluk": "ZOR"
    },
    {
        "soru": "Devamsızlık limiti yüzde kaçtır?",
        "beklenen_kelimeler": ["devamsızlık", "limit", "yüzde", "%"],
        "tip": "spesifik_yuzde",
        "zorluk": "ZOR"
    },
    {
        "soru": "Öğrenci disiplin işlemleri hangi kanun maddesine göre yürütülür?",
        "beklenen_kelimeler": ["54", "madde", "2547", "disiplin"],
        "tip": "spesifik_referans",
        "zorluk": "ZOR"
    },
    {
        "soru": "Mazeret sınavı için başvuru süresi kaç gündür?",
        "beklenen_kelimeler": ["3", "5", "iş günü", "mazeret"],
        "tip": "spesifik_sure",
        "zorluk": "ORTA"
    },
    {
        "soru": "Kayıt yenileme hangi yönetmeliğin kaçıncı maddesinde belirtilmiştir?",
        "beklenen_kelimeler": ["madde", "12", "kayıt", "yenileme"],
        "tip": "spesifik_referans",
        "zorluk": "ORTA"
    },
    {
        "soru": "Yatay geçiş için minimum not ortalaması kaçtır?",
        "beklenen_kelimeler": ["2.50", "3.00", "not", "ortalama"],
        "tip": "spesifik_sayi",
        "zorluk": "ZOR"
    },
    {
        "soru": "Azami öğrenim süresi kaç yıldır?",
        "beklenen_kelimeler": ["yıl", "azami", "süre"],
        "tip": "spesifik_sayi",
        "zorluk": "ORTA"
    },
    {
        "soru": "Genel Sekreter kimdir?",
        "beklenen_kelimeler": ["Genel Sekreter", "Başkan"],
        "tip": "isim_bulma",
        "zorluk": "ÇOK ZOR"
    },
]

def test_question(query_data):
    """Bir soruyu test et"""
    query = query_data["soru"]
    expected_keywords = query_data["beklenen_kelimeler"]
    question_type = query_data["tip"]
    difficulty = query_data["zorluk"]
    
    print(f"\n{'='*80}")
    print(f"SORU: {query}")
    print(f"TIP: {question_type} | ZORLUK: {difficulty}")
    print(f"{'='*80}")
    
    results = db.similarity_search(query, k=5)
    print(f"Bulunan {len(results)} sonuç:\n")
    
    found_relevant = False
    found_keywords = []
    
    for i, doc in enumerate(results, 1):
        source_path = doc.metadata.get('source', '') if doc.metadata.get('source') else ''
        source_name = source_path.split('\\')[-1] if source_path else 'Bilinmiyor'
        page = doc.metadata.get('page', '?')
        content_preview = doc.page_content[:200].replace('\n', ' ')
        
        # Beklenen kelimeler var mı kontrol et
        content_lower = doc.page_content.lower()
        doc_keywords = []
        for keyword in expected_keywords:
            if keyword.lower() in content_lower:
                doc_keywords.append(keyword)
                found_keywords.append(keyword)
        
        marker = "[ILGILI]" if doc_keywords else "[ILGISIZ]"
        if doc_keywords:
            found_relevant = True
        
        print(f"{i}. {source_name} - Sayfa {page} {marker}")
        if doc_keywords:
            print(f"   Bulunan kelimeler: {', '.join(doc_keywords)}")
        print(f"   İçerik: {content_preview}...")
        print()
    
    # Sonuç değerlendirme
    print(f"{'='*80}")
    if found_relevant:
        missing_keywords = set(expected_keywords) - set([k.lower() for k in found_keywords])
        if missing_keywords:
            print(f"[BAŞARILI KISMI] Bulunan kelimeler: {', '.join(set(found_keywords))}")
            print(f"[EKSIK] Bulunamayan kelimeler: {', '.join(missing_keywords)}")
        else:
            print(f"[BAŞARILI] Tüm beklenen kelimeler bulundu: {', '.join(set(found_keywords))}")
    else:
        print(f"[BAŞARISIZ] Beklenen kelimeler bulunamadı: {', '.join(expected_keywords)}")
    print(f"{'='*80}\n")
    
    return found_relevant, len(set(found_keywords)) / len(expected_keywords) if expected_keywords else 0

if __name__ == "__main__":
    print("=" * 80)
    print("ZOR TEST SORULARI - PDF'LERDEN ÇIKARILAN SPESİFİK BİLGİLER")
    print("=" * 80)
    print(f"\nToplam {len(CRITICAL_QUESTIONS)} kritik soru test edilecek.\n")
    
    results = []
    for i, question_data in enumerate(CRITICAL_QUESTIONS, 1):
        print(f"\n[{i}/{len(CRITICAL_QUESTIONS)}]")
        found, match_ratio = test_question(question_data)
        results.append({
            "soru": question_data["soru"],
            "bulundu": found,
            "eslesme_orani": match_ratio,
            "zorluk": question_data["zorluk"]
        })
    
    # Özet
    print("\n" + "=" * 80)
    print("TEST SONUÇLARI ÖZETİ")
    print("=" * 80)
    
    basarili = sum(1 for r in results if r["bulundu"])
    print(f"\nBaşarılı: {basarili}/{len(results)} ({basarili/len(results)*100:.1f}%)")
    
    print("\nDetaylı Sonuçlar:")
    for r in results:
        durum = "[BAŞARILI]" if r["bulundu"] else "[BAŞARISIZ]"
        print(f"{durum} ({r['zorluk']}) Eşleşme: {r['eslesme_orani']*100:.0f}% - {r['soru'][:60]}...")
    
    print("\n" + "=" * 80)
    print("KRİTİK DEĞERLENDİRME")
    print("=" * 80)
    
    cok_zor_basarili = sum(1 for r in results if r["zorluk"] == "ÇOK ZOR" and r["bulundu"])
    cok_zor_toplam = sum(1 for r in results if r["zorluk"] == "ÇOK ZOR")
    
    print(f"\nÇok Zor Sorular: {cok_zor_basarili}/{cok_zor_toplam}")
    print(f"Zor Sorular: {sum(1 for r in results if r['zorluk'] == 'ZOR' and r['bulundu'])}/{sum(1 for r in results if r['zorluk'] == 'ZOR')}")
    print(f"Orta Sorular: {sum(1 for r in results if r['zorluk'] == 'ORTA' and r['bulundu'])}/{sum(1 for r in results if r['zorluk'] == 'ORTA')}")
    
    if cok_zor_basarili == cok_zor_toplam:
        print("\n🎉 TEBRİKLER! Tüm çok zor sorular başarıyla cevaplandı!")
    elif basarili >= len(results) * 0.7:
        print("\n✅ İyi! %70+ başarı oranı var.")
    else:
        print("\n⚠️ İyileştirme gerekli. Başarı oranı %70'in altında.")


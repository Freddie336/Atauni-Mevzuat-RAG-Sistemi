# İsim Araması İyileştirmeleri

## Yapılan İyileştirmeler

### 1. Chunk Size Optimizasyonu
- **Önceki**: 1000 karakter
- **Yeni**: 500 karakter
- **Neden**: İsimler gibi küçük bilgiler için daha iyi precision

### 2. İsim Araması Tespiti
- "kim" sorusu veya büyük harfle başlayan kelimeler tespit ediliyor
- İsim aramaları için query expansion yapılmıyor (exact match öncelikli)

### 3. Exact Match Öncelikli Arama
- İsim aramaları için hybrid search bypass edildi
- Keyword search ile exact match öncelikli
- Case-sensitive exact match bonus: 0.8
- Case-insensitive exact match bonus: 0.5

### 4. Fallback Mekanizması
- LLM cevap vermezse direkt dokümanlardan bilgi çıkarılıyor
- İsimin geçtiği satır ve çevresindeki bilgiler gösteriliyor
- Kaynak ve sayfa numarası belirtiliyor

## Önemli: Vector Store Yeniden Oluşturma

⚠️ **Chunk size değiştiği için vector store'u yeniden oluşturmanız gerekiyor:**

```bash
python scripts/create_memory_for_llm.py
```

## Beklenen Sonuçlar

"Batıkan AKSOY kim" sorusu için:
1. İsim bulunacak (exact match öncelikli)
2. LLM cevap verirse: "Başkan a., Genel Sekreter" bilgisi
3. LLM cevap vermezse: Fallback mekanizması devreye girecek ve dokümanlardan direkt bilgi çıkarılacak

## Test

Vector store'u yeniden oluşturduktan sonra test edin:
- "Batıkan AKSOY kim"
- "Batıkan AKSOY"
- "AKSOY kimdir"

Sonuçlar daha iyi olmalı!


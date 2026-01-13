# Chunk Size Optimizasyonu

## Değişiklikler

### Önceki Ayarlar
- **CHUNK_SIZE**: 1000 karakter
- **CHUNK_OVERLAP**: 200 karakter

### Yeni Ayarlar
- **CHUNK_SIZE**: 500 karakter (50% azaltıldı)
- **CHUNK_OVERLAP**: 100 karakter (50% azaltıldı)

## Neden Küçültüldü?

1. **Daha İyi Precision**: Küçük chunk'lar daha spesifik bilgiler içerir
2. **İsim Aramaları**: İsimler gibi küçük bilgiler için daha iyi sonuçlar
3. **Daha Az Gürültü**: Büyük chunk'lar gereksiz bilgi içerebilir
4. **Daha Hızlı Retrieval**: Küçük chunk'lar daha hızlı işlenir

## Vector Store Yeniden Oluşturma Gerekiyor

⚠️ **ÖNEMLİ**: Chunk size değiştiği için vector store'u yeniden oluşturmanız gerekiyor:

```bash
python scripts/create_memory_for_llm.py
```

## Beklenen Sonuçlar

- Daha fazla chunk sayısı (yaklaşık 2x)
- Daha iyi isim araması sonuçları
- Daha spesifik cevaplar
- Daha az gereksiz bilgi

## İsim Araması İyileştirmeleri

1. İsim aramaları için hybrid search bypass edildi
2. Exact match öncelikli keyword search kullanılıyor
3. 2x daha fazla doküman taranıyor
4. Case-sensitive exact match bonus eklendi


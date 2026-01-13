# 🎓 Üniversite Yönetmelik Asistanı - RAG-Based University Regulation Question Answering System

Üniversite yönetmelikleri soru-cevap sistemi. RAG (Retrieval Augmented Generation) mimarisi kullanarak yönetmelik dokümanlarından bilgi çekerek sorularınızı yanıtlar.

## 📋 Özellikler

- ✅ **RAG Mimarisi**: FAISS vector store ile doküman tabanlı bilgi erişimi
- ✅ **Hybrid Search**: Keyword + semantic search kombinasyonu
- ✅ **Halüsinasyon Kontrolü**: Post-processing ile cevap doğrulama
- ✅ **Confidence Score**: Cevap güvenilirlik skoru
- ✅ **Re-ranking**: Cross-encoder ile doküman yeniden sıralama
- ✅ **Inline Citations**: Cevapta kaynak gösterimi
- ✅ **Performance Tracking**: Detaylı performans metrikleri
- ✅ **Retrieval Cache**: Hızlı yanıt için cache sistemi
- ✅ **Otomatik Eş Anlamlı Sistemi**: Dictionary-based query expansion

## 🚀 Kurulum

### Gereksinimler

- Python 3.8+
- Groq API Key

### Adımlar

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd universite-yonetmelik-bot
```

2. **Virtual environment oluşturun:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# veya
source venv/bin/activate  # Linux/Mac
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Environment değişkenlerini ayarlayın:**
```bash
# .env dosyası oluşturun
GROQ_API_KEY=your-api-key-here
```

5. **Vector store'u oluşturun:**
```bash
python scripts/create_memory_for_llm.py
```

6. **Uygulamayı çalıştırın:**
```bash
streamlit run src/app.py
# veya
python run.py
```

## 📁 Proje Yapısı

```
universite-yonetmelik-bot/
├── src/                    # Ana uygulama kodu
│   ├── app.py             # Streamlit uygulaması
│   ├── config.py          # Konfigürasyon
│   ├── models/            # Model sınıfları
│   ├── utils/             # Yardımcı fonksiyonlar
│   └── services/          # Servisler
├── tests/                 # Test dosyaları
├── docs/                  # Dokümantasyon
├── scripts/               # Yardımcı scriptler
├── data/                  # Dokümanlar (PDF, TXT)
├── vectorstore/           # FAISS vector store
└── logs/                  # Log dosyaları
```

## 🔧 Konfigürasyon

Tüm ayarlar `src/config.py` dosyasında yapılır:

- Retrieval ayarları (MMR, k değerleri)
- LLM ayarları (model, temperature)
- Cache ayarları
- Citation ayarları
- Performance tracking ayarları

## 📖 Kullanım

1. Streamlit uygulamasını başlatın
2. Sol tarafta "RAG Kullanmadan" ve sağ tarafta "RAG Kullanarak" cevapları görürsünüz
3. Üniversite yönetmelikleri hakkında sorularınızı sorun
4. Sidebar'dan performans istatistiklerini görüntüleyin

## 🧪 Test

```bash
# Tüm testleri çalıştır
pytest tests/

# Belirli bir test
pytest tests/test_retrieval.py
```

## 📚 Dokümantasyon

Detaylı dokümantasyon için `docs/` klasörüne bakın:

- `docs/reports/` - Analiz raporları
- `docs/guides/` - Kullanım rehberleri

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Lisans

Bu proje eğitim amaçlıdır.

## 👥 Yazar

Üniversite Yönetmelik Bot Team

## 🙏 Teşekkürler

- LangChain
- Streamlit
- Groq
- HuggingFace


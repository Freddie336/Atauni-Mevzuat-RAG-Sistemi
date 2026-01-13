"""
PDF'lere göre test soruları - Üniversite Yönetmelikleri
Bu sorular PDF'lerdeki bilgilere göre cevap vermeli
"""

# PDF'lere göre hazırlanmış örnek sorular
TEST_QUESTIONS = [
    # Öğrenci İşleri ve Kayıt
    "Kayıt yenileme nasıl yapılır?",
    "Devamsızlık limiti nedir?",
    "Mazeret sınavı için ne yapmalıyım?",
    "Kayıt dondurma işlemi nasıl yapılır?",
    
    # Akademik İşler
    "Staj süresi ne kadar?",
    "Mezuniyet için kaç kredi gerekli?",
    "Ders tekrarı kuralları neler?",
    "Yaz okulu nedir ve nasıl başvurulur?",
    "Ders muafiyeti nasıl yapılır?",
    
    # Yatay Geçiş ve Transfer
    "Yatay geçiş şartları neler?",
    "Kurumlar arası geçiş nasıl yapılır?",
    "Kredi transferi nasıl olur?",
    
    # Çift Anadal ve Yan Dal
    "Çift anadal programı nedir?",
    "Yan dal programına nasıl başvurulur?",
    "Çift anadal için şartlar neler?",
    
    # Disiplin ve Cezalar
    "Disiplin cezaları nelerdir?",
    "Öğrenci disiplin işlemleri nasıl yürütülür?",
    
    # Burs ve Finansal
    "Burs başvurusu nasıl yapılır?",
    "Harç ücretleri ne kadar?",
    
    # Yabancı Dil
    "Yabancı dil hazırlık sınıfı nedir?",
    "Yabancı dil muafiyeti nasıl alınır?",
    "Zorunlu yabancı dil dersleri neler?",
    
    # Uzaktan Eğitim
    "Uzaktan eğitim programlarına nasıl başvurulur?",
    "Açık öğretim fakültesi nedir?",
    
    # Diploma ve Mezuniyet
    "Diploma nasıl alınır?",
    "Mezuniyet belgesi için ne gerekli?",
    "Transkript nasıl alınır?",
    
    # Öğrenci Konseyi
    "Öğrenci konseyi nedir?",
    "Öğrenci konseyi seçimleri nasıl yapılır?",
    
    # Değişim Programları
    "Erasmus programı nedir?",
    "Mevlana değişim programı nasıl başvurulur?",
    "Öğrenci değişim programları neler?",
    
    # Genel Sorular
    "Öğrenci danışmanlığı nedir?",
    "Akademik takvim nedir?",
    "Sınav sistemi nasıl çalışır?",
    "Başarı değerlendirme sistemi nedir?",
    
    # Spesifik Yönetmelik Soruları
    "2547 sayılı kanunun 44. maddesi nedir?",
    "Öğrenci disiplin yönetmeliği nedir?",
    "Lisansüstü eğitim yönetmeliği nedir?",
]

# Kategorilere göre gruplandırılmış sorular
CATEGORIZED_QUESTIONS = {
    "Öğrenci İşleri": [
        "Kayıt yenileme nasıl yapılır?",
        "Devamsızlık limiti nedir?",
        "Mazeret sınavı için ne yapmalıyım?",
        "Kayıt dondurma işlemi nasıl yapılır?",
    ],
    "Akademik İşler": [
        "Staj süresi ne kadar?",
        "Mezuniyet için kaç kredi gerekli?",
        "Ders tekrarı kuralları neler?",
        "Yaz okulu nedir ve nasıl başvurulur?",
        "Ders muafiyeti nasıl yapılır?",
    ],
    "Yatay Geçiş": [
        "Yatay geçiş şartları neler?",
        "Kurumlar arası geçiş nasıl yapılır?",
        "Kredi transferi nasıl olur?",
    ],
    "Çift Anadal/Yan Dal": [
        "Çift anadal programı nedir?",
        "Yan dal programına nasıl başvurulur?",
        "Çift anadal için şartlar neler?",
    ],
    "Disiplin": [
        "Disiplin cezaları nelerdir?",
        "Öğrenci disiplin işlemleri nasıl yürütülür?",
    ],
    "Yabancı Dil": [
        "Yabancı dil hazırlık sınıfı nedir?",
        "Yabancı dil muafiyeti nasıl alınır?",
        "Zorunlu yabancı dil dersleri neler?",
    ],
    "Diploma/Mezuniyet": [
        "Diploma nasıl alınır?",
        "Mezuniyet belgesi için ne gerekli?",
        "Transkript nasıl alınır?",
    ],
    "Değişim Programları": [
        "Erasmus programı nedir?",
        "Mevlana değişim programı nasıl başvurulur?",
        "Öğrenci değişim programları neler?",
    ],
}

if __name__ == "__main__":
    print("=" * 80)
    print("PDF'LERE GÖRE TEST SORULARI - ÜNİVERSİTE YÖNETMELİKLERİ")
    print("=" * 80)
    print(f"\nToplam {len(TEST_QUESTIONS)} soru hazırlandı.\n")
    
    print("\nKategorilere Gore Sorular:\n")
    for category, questions in CATEGORIZED_QUESTIONS.items():
        print(f"\n{category}:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
    
    print("\n\n" + "=" * 80)
    print("Tüm Sorular Listesi:")
    print("=" * 80)
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"{i:2d}. {question}")


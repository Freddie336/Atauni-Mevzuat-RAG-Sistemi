"""
Sistem testi - PDF'lere göre soruları test et
"""
import sys
from pathlib import Path

# Test soruları
TEST_QUESTIONS = [
    "Staj süresi ne kadar?",
    "Devamsızlık limiti nedir?",
    "Yatay geçiş şartları neler?",
    "Çift anadal programı nedir?",
    "Yaz okulu nedir ve nasıl başvurulur?",
    "Mazeret sınavı için ne yapmalıyım?",
    "Mezuniyet için kaç kredi gerekli?",
    "Disiplin cezaları nelerdir?",
    "Kayıt yenileme nasıl yapılır?",
    "Ders muafiyeti nasıl yapılır?",
]

def check_vectorstore():
    """Vector store'un var olup olmadığını kontrol et"""
    vectorstore_path = Path("vectorstore/db_faiss")
    index_faiss = vectorstore_path / "index.faiss"
    index_pkl = vectorstore_path / "index.pkl"
    
    if index_faiss.exists() and index_pkl.exists():
        print("[OK] Vector store mevcut")
        return True
    else:
        print("[HATA] Vector store bulunamadi!")
        print(f"  Beklenen konum: {vectorstore_path}")
        print("\nVector store'u oluşturmak için:")
        print("  python scripts/create_memory_for_llm.py")
        return False

def check_env():
    """Environment değişkenlerini kontrol et"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        print("[OK] GROQ_API_KEY bulundu")
        return True
    else:
        print("[HATA] GROQ_API_KEY bulunamadi!")
        print("\n.env dosyası oluşturun ve şunu ekleyin:")
        print("  GROQ_API_KEY=your-api-key-here")
        return False

def main():
    print("=" * 80)
    print("SISTEM TESTI - PDF'LERE GORE SORULAR")
    print("=" * 80)
    print()
    
    # Kontroller
    print("1. Vector Store Kontrolu:")
    has_vectorstore = check_vectorstore()
    print()
    
    print("2. Environment Degiskenleri:")
    has_env = check_env()
    print()
    
    print("3. Test Sorulari:")
    print(f"   Toplam {len(TEST_QUESTIONS)} soru hazir")
    print()
    
    if has_vectorstore and has_env:
        print("=" * 80)
        print("SISTEM HAZIR - TEST EDEBILIRSINIZ")
        print("=" * 80)
        print("\nTest sorulari:")
        for i, q in enumerate(TEST_QUESTIONS, 1):
            print(f"  {i:2d}. {q}")
        print("\nUygulamayi baslatmak icin:")
        print("  streamlit run src/app.py")
        print("  veya")
        print("  python run.py")
    else:
        print("=" * 80)
        print("SISTEM HAZIR DEGIL - LUTFEN EKSIKLERI TAMAMLAYIN")
        print("=" * 80)

if __name__ == "__main__":
    main()


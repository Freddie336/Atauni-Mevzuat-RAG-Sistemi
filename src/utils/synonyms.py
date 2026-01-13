"""
Otomatik Eş Anlamlı Sistemi - Dictionary-Based Query Expansion
Manuel if-else blokları yerine dictionary kullanarak ölçeklenebilir çözüm
"""

import json
from pathlib import Path
from typing import Dict, List

# Eş anlamlı sözlüğü
MEDICAL_SYNONYMS = {
    "ibs": ["IBS", "İBS", "irritabl bağırsak sendromu", "irritable bowel syndrome", "huzursuz bağırsak sendromu", "IBS hastalığı", "IBS belirtileri", "IBS nedir"],
    "irritabl": ["IBS", "irritabl bağırsak sendromu", "irritable bowel syndrome", "huzursuz bağırsak", "irritabl bağırsak sendromu belirtileri"],
    "huzursuz bağırsak": ["IBS", "irritabl bağırsak sendromu", "irritable bowel syndrome", "huzursuz bağırsak sendromu belirtileri"],
    "irritabl bağırsak sendromu": ["IBS", "İBS", "irritabl bağırsak sendromu belirtileri", "IBS belirtileri"],
    
    "reflü": ["GERD", "gastroözofageal reflü", "mide yanması", "mide ekşimesi", "heartburn", "reflü sebepleri", "reflü nedenleri"],
    "mide yanması": ["reflü", "GERD", "gastroözofageal reflü", "mide ekşimesi", "heartburn"],
    "mide ekşimesi": ["reflü", "GERD", "gastroözofageal reflü", "mide yanması"],
    
    "kanser": ["cancer", "kanser hastalığı", "malignite", "tümör", "kanser nedir", "kanser hakkında"],
    
    "diyabet": ["diabetes", "şeker hastalığı", "glikoz", "tip2 diyabet", "tip 2 diyabet"],
    "şeker hastalığı": ["diyabet", "diabetes", "tip2 diyabet"],
    
    "hipertansiyon": ["yüksek tansiyon", "hypertension", "HT"],
    "yüksek tansiyon": ["hipertansiyon", "hypertension"],
    
    "kalp yetmezliği": ["heart failure", "cardiac failure", "konjestif kalp yetmezliği", "kalp yetersizliği", "kalp yetmezliği tedavisi", "kalp yetmezliği nedir"],
    "kalp yetersizliği": ["kalp yetmezliği", "heart failure", "cardiac failure", "konjestif kalp yetmezliği"],
    "heart failure": ["kalp yetmezliği", "cardiac failure", "konjestif kalp yetmezliği"],
    
    "gastrit": ["mide iltihabı", "gastritis"],
    "ülser": ["peptik ülser", "ulcer"],
    "anemi": ["kansızlık", "anemia"],
    "tiroit": ["tiroid", "thyroid"],
    "tiroid": ["tiroit", "thyroid"],
    
    "astım": ["asthma", "nefes darlığı", "astım nedir", "astım belirtileri", "astım tedavisi", "astım hastalığı", "bronşiyal astım"],
    "asthma": ["astım", "nefes darlığı", "astım nedir", "astım belirtileri", "astım tedavisi"],
    "bronşit": ["bronchitis", "bronş iltihabı"],
    "pnömoni": ["zatürre", "pneumonia"],
    "zatürre": ["pnömoni", "pneumonia"],
    
    "farenjit": ["boğaz iltihabı", "pharyngitis", "boğaz ağrısı", "sore throat"],
    "boğaz ağrısı": ["farenjit", "pharyngitis", "tonsillit", "sore throat"],
    
    "sinüzit": ["sinusitis", "sinüs iltihabı"],
    
    "grip": ["influenza", "soğuk algınlığı", "nezle", "üst solunum yolu enfeksiyonları", "viral enfeksiyon", "grip tedavisi", "grip nasıl geçer"],
    "influenza": ["grip", "soğuk algınlığı", "nezle", "üst solunum yolu enfeksiyonları"],
    "soğuk algınlığı": ["grip", "influenza", "nezle", "üst solunum yolu enfeksiyonları"],
    "nezle": ["grip", "influenza", "soğuk algınlığı", "üst solunum yolu enfeksiyonları"],
    "otit": ["kulak iltihabı", "otitis"],
    "konjonktivit": ["göz iltihabı", "conjunctivitis"],
    "dermatit": ["egzama", "eczema", "dermatitis"],
    "egzama": ["dermatit", "eczema", "dermatitis"],
    
    "artrit": ["eklem iltihabı", "arthritis"],
    "osteoporoz": ["kemik erimesi", "osteoporosis"],
    "kemik erimesi": ["osteoporoz", "osteoporosis"],
    
    "migren": ["baş ağrısı", "headache", "migraine"],
    "baş ağrısı": ["migren", "headache", "migraine"],
    
    "bel ağrısı": ["boyun bel ağrıları", "omurga ağrı", "spinal pain", "lumbago"],
    "fıtık": ["fitik", "hernia", "herni"],
    "fitik": ["fıtık", "hernia", "herni"],
    
    "kemik kırılması": ["kırık", "fracture", "bone break", "kemik kırığı"],
    "kırık": ["kemik kırılması", "fracture", "bone break"],
    
    "burun akıntısı": ["rinore", "nazal akıntı", "burun akıntısı sebepleri", "burun akıntısı nedenleri", "burun akıntısı etkenleri"],
    "rinore": ["burun akıntısı", "nazal akıntı", "rinore sebepleri", "rinore nedenleri"],
    "nazal akıntı": ["burun akıntısı", "rinore", "nazal akıntı sebepleri"],
    "vitamin": ["vitamin eksikliği", "vitamin kaynakları", "vitamin fonksiyonu"],
    
    "sebep": ["neden", "etken", "faktör", "kaynak"],
    "neden": ["sebep", "etken", "faktör", "kaynak"],
    
    "tedavi": ["ilaç", "önlem", "iyi gelir", "çözüm"],
    "ilaç": ["tedavi", "önlem"],
    "ne iyi gelir": ["tedavi", "ilaç", "önlem"],
    
    "ne yapmam lazım": ["tedavi", "ilaç", "önlem", "ne yapmalı"],
    "ne yapmalı": ["tedavi", "ilaç", "önlem"],
}


def load_synonym_dict(file_path: str = None) -> Dict[str, List[str]]:
    """
    Eş anlamlı sözlüğünü yükler.
    
    Args:
        file_path: JSON dosya yolu (opsiyonel)
        
    Returns:
        Dict: Eş anlamlı sözlüğü
    """
    if file_path and Path(file_path).exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Synonym dictionary yüklenemedi: {e}")
            return MEDICAL_SYNONYMS
    return MEDICAL_SYNONYMS


def expand_query_with_dictionary(query: str, synonym_dict: Dict[str, List[str]] = None) -> str:
    """
    Dictionary-based query expansion - Otomatik eş anlamlı ekleme
    
    Args:
        query: Kullanıcı sorgusu
        synonym_dict: Eş anlamlı sözlüğü (opsiyonel)
        
    Returns:
        str: Genişletilmiş sorgu
    """
    if synonym_dict is None:
        synonym_dict = MEDICAL_SYNONYMS
    
    query_lower = query.lower()
    expanded_terms = [query]  # Orijinal sorguyu da ekle
    
    # Sözlükte ara ve eş anlamlıları ekle
    for key, synonyms in synonym_dict.items():
        if key in query_lower:
            expanded_terms.extend(synonyms)
    
    # Benzersiz terimleri döndür
    return " ".join(set(expanded_terms))


def save_synonym_dict(synonym_dict: Dict[str, List[str]], file_path: str = "medical_synonyms.json"):
    """
    Eş anlamlı sözlüğünü JSON dosyasına kaydeder.
    
    Args:
        synonym_dict: Eş anlamlı sözlüğü
        file_path: Kayıt edilecek dosya yolu
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(synonym_dict, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Test
    test_queries = [
        "IBS hastalığı ne",
        "reflü sebepleri",
        "kanser nedir",
        "bel ağrısına ne iyi gelir"
    ]
    
    for query in test_queries:
        expanded = expand_query_with_dictionary(query)
        print(f"Orijinal: {query}")
        print(f"Genişletilmiş: {expanded}")
        print()


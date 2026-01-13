"""
Streamlit uygulamasını çalıştırmak için ana dosya.
"""
import sys
from pathlib import Path

# Proje root'unu path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # Streamlit'i src/app.py ile çalıştır
    sys.argv = ["streamlit", "run", str(project_root / "src" / "app.py")]
    sys.exit(stcli.main())


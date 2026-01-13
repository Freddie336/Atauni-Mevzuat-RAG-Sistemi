"""
Üniversite Yönetmelik Asistanı - Setup dosyası
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = []
    for line in fh:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("-i"):
            # Platform-specific dependencies'leri atla (python_version, platform_machine vb.)
            if ";" in line:
                # "package==version; condition" formatından sadece "package==version" kısmını al
                package_part = line.split(";")[0].strip()
                if package_part:
                    requirements.append(package_part)
            else:
                requirements.append(line)

setup(
    name="universite-yonetmelik-bot",
    version="1.0.0",
    author="Üniversite Yönetmelik Bot Team",
    description="RAG-based University Regulation Chatbot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "universite-yonetmelik-bot=run:main",
        ],
    },
)


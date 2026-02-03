# Poređenje Decision Tree i Random Forest algoritama

Seminarski rad iz predmeta **Mašinsko učenje** - Master studije

## 📋 Opis projekta

Projekat demonstrira poređenje dva popularna algoritma za klasifikaciju:
- **Decision Tree** (Stablo odlučivanja)
- **Random Forest** (Nasumična šuma)

Algoritmi su testirani na poznatom **Iris datasetu** koji sadrži 150 uzoraka cvetova irisa sa 4 atributa.

## 📁 Struktura projekta

```
MasinskoUcenje/
├── Iris.csv                    # Dataset
├── poredjenje_seminarski.py    # Glavni Python skript
├── slike/                      # Generisani grafici
├── venv/                       # Virtuelno okruženje (ne uključeno u repo)
└── README.md                   # Ovaj fajl
```

## 🚀 Pokretanje projekta

### 1. Kreiranje virtuelnog okruženja

**macOS / Linux:**
```bash
python3 -m venv venv
```

**Windows:**
```bash
python -m venv venv
```

### 2. Aktiviranje virtuelnog okruženja

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

### 3. Instalacija zavisnosti

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Pokretanje skripte

**macOS / Linux:**
```bash
python3 poredjenje_seminarski.py
```

**Windows:**
```bash
python poredjenje_seminarski.py
```

## 📊 Rezultati

Skripta generiše:
- Uporednu analizu tačnosti oba modela
- 5-fold Cross-Validation rezultate
- Matrice konfuzije
- Vizualizacije stabala odlučivanja
- Grafike važnosti atributa

Svi grafici se čuvaju u folderu `slike/`.

## 📦 Zavisnosti

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 📈 Rezultati evaluacije

| Model | Test tačnost | Cross-Validation |
|-------|-------------|------------------|
| Decision Tree | 93.33% | 95.33% (±3.40%) |
| Random Forest | 90.00% | 96.67% (±2.11%) |

## 👤 Autor

Lazar Birtašević M17/2025 - Predmet: Mašinsko učenje (prof. Marija Mojsilović)
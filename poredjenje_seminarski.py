import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Kreiranje foldera za slike
SLIKE_FOLDER = 'slike'
if not os.path.exists(SLIKE_FOLDER):
    os.makedirs(SLIKE_FOLDER)

# 1. UČITAVANJE I PREGLED PODATAKA
print("=" * 60)
print("POREĐENJE DECISION TREE I RANDOM FOREST ALGORITAMA")
print("=" * 60)

# Učitavanje dataseta
df = pd.read_csv('Iris.csv')

# Osnovni pregled
print("\n1. PREGLED DATASETA")
print("-" * 40)
print(f"Broj uzoraka: {df.shape[0]}")
print(f"Broj atributa: {df.shape[1] - 2}")  # bez Id i Species
print(f"Nedostajuće vrednosti: {df.isnull().sum().sum()}")
print(f"\nDistribucija klasa:\n{df['Species'].value_counts()}")

# 2. PRIPREMA PODATAKA
print("\n2. PRIPREMA PODATAKA")
print("-" * 40)

# Definisanje atributa i ciljne varijable
atributi = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[atributi]
y = df['Species']

# Enkodiranje klasa
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Podela na trening (80%) i test (20%) skup
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Trening skup: {len(X_train)} uzoraka")
print(f"Test skup: {len(X_test)} uzoraka")

# 3. TRENIRANJE MODELA
print("\n3. TRENIRANJE MODELA")
print("-" * 40)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Oba modela su uspešno trenirana.")

# 4. EVALUACIJA - TAČNOST
print("\n4. REZULTATI NA TEST SKUPU")
print("-" * 40)

acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"Decision Tree tačnost: {acc_dt:.2%}")
print(f"Random Forest tačnost: {acc_rf:.2%}")

# 5. CROSS-VALIDATION (pouzdanija evaluacija)
print("\n5. 5-FOLD CROSS-VALIDATION")
print("-" * 40)

cv_dt = cross_val_score(dt_model, X, y, cv=5)
cv_rf = cross_val_score(rf_model, X, y, cv=5)

print(f"Decision Tree: {cv_dt.mean():.2%} (±{cv_dt.std():.2%})")
print(f"Random Forest: {cv_rf.mean():.2%} (±{cv_rf.std():.2%})")

# 6. MATRICE KONFUZIJE
print("\n6. MATRICE KONFUZIJE")
print("-" * 40)

klase = encoder.classes_

print("\nDecision Tree:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nRandom Forest:")
print(confusion_matrix(y_test, y_pred_rf))

# 7. VAŽNOST ATRIBUTA
print("\n7. VAŽNOST ATRIBUTA")
print("-" * 40)

print(f"\n{'Atribut':<20} {'Decision Tree':>15} {'Random Forest':>15}")
print("-" * 50)
for i, attr in enumerate(atributi):
    print(f"{attr:<20} {dt_model.feature_importances_[i]:>15.3f} {rf_model.feature_importances_[i]:>15.3f}")

# 8. VIZUALIZACIJE

# Grafik 1: Poređenje tačnosti - stupčasti grafik
fig, ax = plt.subplots(figsize=(10, 6))
modeli = ['Decision Tree', 'Random Forest']
x = np.arange(len(modeli))
width = 0.35

# Test tačnost i CV tačnost uporedo
test_acc = [acc_dt, acc_rf]
cv_acc = [cv_dt.mean(), cv_rf.mean()]

bars1 = ax.bar(x - width/2, test_acc, width, label='Test skup', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, cv_acc, width, label='Cross-Validation', color='#2ecc71', edgecolor='black')

ax.set_ylabel('Tačnost')
ax.set_title('Poređenje tačnosti: Test skup vs Cross-Validation')
ax.set_xticks(x)
ax.set_xticklabels(modeli)
ax.set_ylim([0.85, 1.0])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Dodavanje vrednosti
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'{bar.get_height():.1%}', ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'{bar.get_height():.1%}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SLIKE_FOLDER, 'poredjenje_tacnosti.png'), dpi=150)
plt.show()

# Grafik 2: Cross-Validation rezultati po foldovima
fig, ax = plt.subplots(figsize=(10, 6))
foldovi = np.arange(1, 6)
ax.plot(foldovi, cv_dt, 'o-', color='#3498db', linewidth=2, markersize=8, label=f'Decision Tree (prosek: {cv_dt.mean():.1%})')
ax.plot(foldovi, cv_rf, 's-', color='#2ecc71', linewidth=2, markersize=8, label=f'Random Forest (prosek: {cv_rf.mean():.1%})')
ax.axhline(y=cv_dt.mean(), color='#3498db', linestyle='--', alpha=0.5)
ax.axhline(y=cv_rf.mean(), color='#2ecc71', linestyle='--', alpha=0.5)
ax.set_xlabel('Fold')
ax.set_ylabel('Tačnost')
ax.set_title('5-Fold Cross-Validation - Rezultati po foldovima')
ax.set_xticks(foldovi)
ax.set_ylim([0.85, 1.02])
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SLIKE_FOLDER, 'cross_validation.png'), dpi=150)
plt.show()

# Grafik 3: Matrice konfuzije
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', 
            cmap='Blues', xticklabels=klase, yticklabels=klase, ax=axes[0])
axes[0].set_title('Decision Tree')
axes[0].set_xlabel('Predviđeno')
axes[0].set_ylabel('Stvarno')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', 
            cmap='Greens', xticklabels=klase, yticklabels=klase, ax=axes[1])
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Predviđeno')
axes[1].set_ylabel('Stvarno')

plt.suptitle('Matrice konfuzije', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SLIKE_FOLDER, 'matrice_konfuzije.png'), dpi=150)
plt.show()

# Grafik 4: Vizualizacija Decision Tree stabla
plt.figure(figsize=(16, 10))
plot_tree(dt_model, feature_names=atributi, class_names=list(klase),
          filled=True, rounded=True, fontsize=9)
plt.title('Decision Tree - Struktura stabla')
plt.tight_layout()
plt.savefig(os.path.join(SLIKE_FOLDER, 'decision_tree_stablo.png'), dpi=150)
plt.show()

# Grafik 5: Vizualizacija jednog stabla iz Random Forest-a
plt.figure(figsize=(16, 10))
plot_tree(rf_model.estimators_[0], feature_names=atributi, class_names=list(klase),
          filled=True, rounded=True, fontsize=9, max_depth=4)
plt.title('Random Forest - Jedno stablo iz šume (od ukupno 100)')
plt.tight_layout()
plt.savefig(os.path.join(SLIKE_FOLDER, 'random_forest_stablo.png'), dpi=150)
plt.show()

# Grafik 6: Važnost atributa
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(atributi))
width = 0.35
ax.bar(x - width/2, dt_model.feature_importances_, width, label='Decision Tree', color='#3498db', edgecolor='black')
ax.bar(x + width/2, rf_model.feature_importances_, width, label='Random Forest', color='#2ecc71', edgecolor='black')
ax.set_ylabel('Važnost')
ax.set_title('Važnost atributa za klasifikaciju')
ax.set_xticks(x)
ax.set_xticklabels(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SLIKE_FOLDER, 'vaznost_atributa.png'), dpi=150)
plt.show()

# 🌿 Détection des Maladies des Plantes - CNN
## Détection automatique des maladies des plantes à partir d'images de feuilles باستخدام CNN

---

## 📁 Structure du Projet

```
ProjetDL/
│
├── README.md                      ← Ce fichier
├── requirements.txt               ← Dépendances Python
│
├── data/
│   └── (placer ici le dataset PlantVillage)
│
├── 01_charger_dataset.py          ← Étape 2 : Chargement et exploration du dataset
├── 02_preprocessing.py            ← Étape 3 : Prétraitement et augmentation
├── 03_modele_cnn_simple.py        ← Étape 4a : CNN depuis zéro
├── 04_modele_transfer_learning.py ← Étape 4b : MobileNetV2 Transfer Learning
├── 05_entrainement.py             ← Étape 5 : Entraînement complet
├── 06_evaluation.py               ← Étape 6 : Évaluation et métriques
└── 07_deploiement_streamlit.py    ← Étape 8 : Application web Streamlit
```

---

## 🚀 Comment démarrer

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Télécharger le Dataset PlantVillage
- Lien : https://www.kaggle.com/datasets/emmarex/plantdisease


### 3. Lancer les scripts dans l'ordre
```bash
python 01_charger_dataset.py
python 02_preprocessing.py
python 05_entrainement.py
python 06_evaluation.py
```

---

## 📊 Dataset : PlantVillage
- **38 classes** de maladies et plantes saines
- **+50 000 images** de feuilles
- Résolution : 256x256 pixels

---

## 🏆 Résultats attendus
| Modèle              | Accuracy attendue |
|---------------------|------------------|
| CNN Simple          | ~75-85%          |
| MobileNetV2 (TL)    | ~90-97%          |

---

*Projet réalisé dans le cadre du cours Deep Learning - 4EAS2*



































# Plant Disease Detection

## Installation
pip install -r requirements.txt

## Run Training
python 05_entrainement.py

## Test Image
python test_prediction.py

## Run App
streamlit run 07_deploiement_streamlit.py
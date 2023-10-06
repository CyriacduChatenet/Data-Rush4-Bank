import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class Accuracy:
    def calculateAccuracy(data_encoded):
        # Séparer les données en variables indépendantes et dépendantes
        X = data_encoded.iloc[:, :-1].values
        y = data_encoded.iloc[:, -1].values
        
        # Diviser les données en ensembles d'entraînement et de test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Créer un objet de régression logistique
        classifier = LogisticRegression(random_state=0)
        
        # Entraîner le modèle sur l'ensemble d'entraînement
        classifier.fit(X_train, y_train)
        
        # Prédire les résultats sur l'ensemble de test
        y_pred = classifier.predict(X_test)
        
        # Afficher les résultats
        print("Accuracy:", classifier.score(X_test, y_test))
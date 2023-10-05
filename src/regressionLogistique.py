import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class LogisticRegressionAnalysis:

    data = pd.read_csv('data/data.csv')
    
    @staticmethod
    def calculateRegressionLogistique():
        # Séparation des caractéristiques (X) et de la variable cible (Y)
        X = LogisticRegressionAnalysis.data[['month', 'credit_amount', 'credit_term', 'age', 'sex', 'education', 'product_type', 'having_children_flg', 'region', 'income', 'family_status', 'phone_operator', 'is_client']]
        Y = LogisticRegressionAnalysis.data['bad_client_target']

        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Création et ajustement du modèle de régression logistique
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Prédiction sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Rapport de classification
        class_report = classification_report(y_test, y_pred)

        # Tracé de la courbe de régression logistique
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test['month'], y_test, color='red', label='Vraies valeurs')
        plt.scatter(X_test['month'], y_pred, color='blue', label='Prédictions')
        plt.xlabel('Feature1')
        plt.ylabel('Cible')
        plt.legend(loc='upper left')
        plt.title('Courbe de Régression Logistique')
        plt.show()

        # Afficher la matrice de confusion et le rapport de classification
        print("Matrice de Confusion:\n", conf_matrix)
        print("\nRapport de Classification:\n", class_report)
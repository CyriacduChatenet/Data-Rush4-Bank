import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class MyLinearRegression:  # Renommez la classe pour éviter les conflits
    @staticmethod
    def calculateLinearRegression(data_encoded, data):
        # Sélectionner les colonnes nécessaires pour la régression linéaire
        # Vous pouvez ajouter d'autres colonnes si nécessaire
        X = data_encoded[['credit_amount', 'credit_term', 'age', 'income']]
        y = data_encoded['bad_client_target']

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Créer un modèle de régression linéaire
        model = LinearRegression()

        # Entraîner le modèle sur l'ensemble d'entraînement
        model.fit(X_train, y_train)

        # Faire des prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Évaluer les performances du modèle
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Mean Squared Error : {mse}')
        print(f'R-squared : {r2}')

        # Vous pouvez également afficher les coefficients de régression
        print('Coefficients de régression :')
        for i, col in enumerate(X.columns):
            print(f'{col}: {model.coef_[i]}')

        # Pour faire des prédictions sur de nouvelles données, utilisez model.predict()
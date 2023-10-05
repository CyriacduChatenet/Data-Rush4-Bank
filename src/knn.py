def calculateKnn(standardScaler, data_encoded,data):
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    # Normalisation des données
    scaler = standardScaler
    scaled_data = scaler.fit_transform(data_encoded.drop('bad_client_target', axis=1))
    scaled_df = pd.DataFrame(scaled_data, columns=data_encoded.columns[:-1])

    # Séparation des données en ensembles d'entraînement et de test
    X = scaled_df
    y = data['bad_client_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle KNN et sélection de la meilleure valeur de k
    k_values = list(range(1, 51))
    accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    best_k = k_values[accuracies.index(max(accuracies))]

    print(f"Meilleure valeur de k: {best_k} avec une précision de {max(accuracies)*100:.2f}%")
    print("KNN")
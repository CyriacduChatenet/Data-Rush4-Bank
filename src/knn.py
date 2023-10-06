import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

class Knn :
    def create_text_file(filename, content):
        try:
            # Ouvre le fichier en mode écriture (crée un nouveau fichier s'il n'existe pas)
            with open(filename, 'w') as file:
                # Écrit le contenu dans le fichier
                for ligne in content:
                    file.write(ligne + '\n')
            print(f"Fichier '{filename}' créé avec succès.")
        except Exception as e:
            print(f"Erreur lors de la création du fichier '{filename}': {str(e)}")

    def calculateKnn(standardScaler, data_encoded,data):
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

        recalls = []

        precisions = []

        f1_scores = []

    

        for k in k_values:

            knn = KNeighborsClassifier(n_neighbors=k)

            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))

            recalls.append(recall_score(y_test, y_pred, average='binary'))

            precisions.append(precision_score(y_test, y_pred, average='binary'))

            f1_scores.append(f1_score(y_test, y_pred, average='binary'))

    

        best_k = k_values[accuracies.index(max(accuracies))]

    

        # pour une classification binaire

        #recall = recall_score(y_test, y_pred, average='binary')

        # pour une classification multiclasse

        recall = recall_score(y_test, y_pred, average='macro')  

    

    

        print(f"Meilleure valeur de k: {best_k} avec une précision de {max(accuracies)*100:.2f}%")

        print(f"Recall at best k: {recalls[best_k - 1]:.2f}")

        print(f"Precision at best k: {precisions[best_k - 1]:.2f}")

        print(f"F1-score at best k: {f1_scores[best_k - 1]:.2f}")

        print(f"Rappel: {recall:.2f}")

        print("KNN")

        filename = "report.txt"
        content = [f"Meilleure valeur de k: {best_k} avec une précision de {max(accuracies)*100:.2f}%", f"Recall at best k: {recalls[best_k - 1]:.2f}", f"Precision at best k: {precisions[best_k - 1]:.2f}", f"F1-score at best k: {f1_scores[best_k - 1]:.2f}", f"Rappel: {recall:.2f}"]
        Knn.create_text_file(filename, content)

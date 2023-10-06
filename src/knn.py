import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

class Knn:
    @staticmethod
    def create_text_file(filename, content):
        try:
            with open(filename, 'w') as file:
                for ligne in content:
                    file.write(ligne + '\n')
            print(f"Fichier '{filename}' créé avec succès.")
        except Exception as e:
            print(f"Erreur lors de la création du fichier '{filename}': {str(e)}")

    @staticmethod
    def calculateKnn(standardScaler, data_encoded, data, fixed_k=None):
        scaler = standardScaler
        scaled_data = scaler.fit_transform(data_encoded.drop('bad_client_target', axis=1))
        scaled_df = pd.DataFrame(scaled_data, columns=data_encoded.columns[:-1])

        X = scaled_df
        y = data['bad_client_target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if fixed_k is not None and fixed_k >= 1 and fixed_k <= 50:
            k_values = [fixed_k]
        else:
            k_values = list(range(1, 51))

        accuracies = []
        recalls = []
        precisions = []
        f1_scores = []

        best_accuracy = -1  # Initialiser à une valeur négative
        best_k = None

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            recalls.append(recall_score(y_test, y_pred, average='binary'))
            precisions.append(precision_score(y_test, y_pred, average='binary'))
            f1_scores.append(f1_score(y_test, y_pred, average='binary'))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        if best_k is not None:
            recall = recall_score(y_test, y_pred, average='macro')
            print(f"Meilleure valeur de k: {best_k} avec une précision de {best_accuracy*100:.2f}%")
            print(f"Recall at best k: {recalls[k_values.index(best_k)]:.2f}")
            print(f"Precision at best k: {precisions[k_values.index(best_k)]:.2f}")
            print(f"F1-score at best k: {f1_scores[k_values.index(best_k)]:.2f}")
            print(f"Rappel: {recall:.2f}")
            print("KNN")

            filename = "report.txt"
            content = [f"Meilleure valeur de k: {best_k} avec une précision de {best_accuracy*100:.2f}%",
                       f"Recall at best k: {recalls[k_values.index(best_k)]:.2f}",
                       f"Precision at best k: {precisions[k_values.index(best_k)]:.2f}",
                       f"F1-score at best k: {f1_scores[k_values.index(best_k)]:.2f}",
                       f"Rappel: {recall:.2f}"]
            Knn.create_text_file(filename, content)
        else:
            print("Aucune donnée n'est disponible.")

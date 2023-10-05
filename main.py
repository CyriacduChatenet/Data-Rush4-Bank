import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
data = pd.read_csv('data/data.csv')

# Encodage One-Hot des variables catégorielles
categorical_cols = ['sex', 'education', 'product_type', 'family_status']
encoder = OneHotEncoder(drop='first')
encoded_data = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
data_encoded = pd.concat([data.drop(categorical_cols, axis=1), encoded_df], axis=1)

# Normalisation des données
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data_encoded.drop('bad_client_target', axis=1))
# scaled_df = pd.DataFrame(scaled_data, columns=data_encoded.columns[:-1])

correlation_matrix = data_encoded.corr()

plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de Corrélation')
plt.tight_layout()
plt.show()

# Séparation des données en ensembles d'entraînement et de test
# X = scaled_df
# y = data['bad_client_target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Entraînement du modèle KNN et sélection de la meilleure valeur de k
# k_values = list(range(1, 51))
# accuracies = []
# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     accuracies.append(accuracy_score(y_test, y_pred))
# best_k = k_values[accuracies.index(max(accuracies))]

# print(f"Meilleure valeur de k: {best_k} avec une précision de {max(accuracies)*100:.2f}%")

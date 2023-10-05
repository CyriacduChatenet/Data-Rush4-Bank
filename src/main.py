import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import knn
from correlationMatrix import MatriceCorrelation
from regressionLogistique import LogisticRegression

logisticRegression = LogisticRegression

logisticRegression.calculateRegressionLogistique()

# Chargement des données
data = pd.read_csv('../data/data.csv')

# Encodage One-Hot des variables catégorielles
categorical_cols = ['sex', 'education', 'product_type', 'family_status']
encoder = OneHotEncoder(drop='first')
encoded_data = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
data_encoded = pd.concat([data.drop(categorical_cols, axis=1), encoded_df], axis=1)

knn.calculateKnn(StandardScaler(),data_encoded,data)
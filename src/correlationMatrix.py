import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

class MatriceCorrelation :
    def calculateCorrelationMatrix (data_encoded) :
        # matrice de correlation
        correlation_matrix = data_encoded.corr()

        plt.figure(figsize=(20, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matrice de Corrélation')
        plt.tight_layout()
        plt.show()

        print('Correlation Matrix')
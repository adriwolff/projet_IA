#EXPLORATION DES DONNEES

import pandas as pd
import numpy as np

# Chargement des données
data = pd.read_csv('All_Participant_resultat_corrCollisions_delimiter_propositionFinale.csv')

# Affichage des premières lignes du dataframe pour un aperçu
print("Affichage d'un aperçu du dataframe:\n", data.head())

# Affichage des statistiques descriptives
print("Affichage des statistiques descriptives:\n", data.describe())

# Sélection des données du premier essai
data_first_trial = data[data['Essai'] == 1]

# Exclusion de la colonne "ID" et des colonnes non numériques avant de calculer les corrélations
data_numeric = data_first_trial.drop(columns=['ID', 'Proposition_Phase'])

# Visualisation des corrélations
import seaborn as sns
import matplotlib.pyplot as plt

corr = data_numeric.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


#PREPARATION DES DONNEES

# Nettoyage des données
data = data.dropna()  # Suppression des lignes avec des valeurs manquantes

# Sélection des caractéristiques pertinentes
features = data_numeric.drop(columns=["Essai","Nombre erreur"])
target = data_numeric['Nombre erreur']

# Normalisation des données
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


#MODELISATION

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Modèle de régression linéaire
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Dummy Regressor
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

# Évaluation des performances
print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear Regression RMSE:", root_mean_squared_error(y_test, y_pred_lr))
print("Dummy Regressor MAE:", mean_absolute_error(y_test, y_pred_dummy))
print("Dummy Regressor RMSE:", root_mean_squared_error(y_test, y_pred_dummy))

# Prédiction du nombre d'essais nécessaires
y_pred_lr_test = lr.predict(X_test)

# Obtention des index de l'ensemble de test
index_test = data.index[range(len(X_test))]

# Création d'un DataFrame pour les prédictions et les ID correspondants
predictions_df = pd.DataFrame({'ID': data.loc[index_test, 'ID'], 'Nombre Essais Prédit': y_pred_lr_test})

# Affichage des prédictions avec les ID correspondants
print("Prédictions:\n", predictions_df)

# Affichage des prédictions par ID pour le test set
test_indices = data_grouped.iloc[X_test_grouped.index].index
predictions_df = pd.DataFrame({
    'ID': data.loc[test_indices, 'ID'],
    'Nombre Essais Prédit': y_pred_lr_grouped
})

print("Prédictions par ID:")
print(predictions_df.head())


#EVALUATION ET AMELIORATION

# Utilisation de plus de données pour l'amélioration
# Ici nous ajoutons les données des 2 premiers essais
data_grouped = data.groupby('ID').head(2)
features_grouped = data_grouped.drop(columns=['ID', 'Proposition_Phase', 'Temps Experience'])
target_grouped = data_grouped['Temps Experience']

features_grouped_scaled = scaler.fit_transform(features_grouped)

X_train_grouped, X_test_grouped, y_train_grouped, y_test_grouped = train_test_split(features_grouped_scaled, target_grouped, test_size=0.2, random_state=42)

# Modèle de régression linéaire avec les données des 2 premiers essais
lr_grouped = LinearRegression()
lr_grouped.fit(X_train_grouped, y_train_grouped)
y_pred_lr_grouped = lr_grouped.predict(X_test_grouped)

# Évaluation des performances avec les données des 2 premiers essais
print("Linear Regression with first 2 trials MAE:", mean_absolute_error(y_test_grouped, y_pred_lr_grouped))
print("Linear Regression with first 2 trials RMSE:", root_mean_squared_error(y_test_grouped, y_pred_lr_grouped))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as root_mean_squared_error
import math
import numpy as np


# Chargement des données
data = pd.read_csv('All_Participant_resultat_corrCollisions_delimiter_propositionFinale.csv')

# Création d'un DataFrame pour stocker le nombre d'essais nécessaires pour chaque participant pour atteindre 0 erreurs
participants = data['ID'].unique()
essais_necessaires = []

for participant in participants:
    participant_data = data[data['ID'] == participant]
    #print("ID:",participant,"essai: ", len(participant_data['Essai']))
    essais_necessaires.append({'ID': participant, 'Nombre_Essais': len(participant_data['Essai'])})


essais_necessaires_df = pd.DataFrame(essais_necessaires)
print ("Nb d'essais nécessaires par participants:\n", essais_necessaires_df)
plt.hist(essais_necessaires_df['Nombre_Essais'], bins=np.arange(4, 10)-0.5)
plt.xlabel('Nombre d\'essais')
plt.ylabel('Fréquence')
plt.show()

# Sélection des premiers essais
data_grouped = data.groupby('ID').head(1)
    
# Sélection des caractéristiques pertinentes
features = data_grouped.drop(columns=['ID', 'Proposition_Phase', 'Essai'])
target = essais_necessaires_df['Nombre_Essais']

# Normalisation des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

error=[]
for i in range(60):
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=i)

    # Modèle de régression linéaire
    lr = SVR()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Dummy Regressor
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    # Calcul des écarts entre les prédictions et les valeurs réelles
    errors_lr = abs(y_test - y_pred_lr)

    # Calcul de la moyenne des écarts
    mean_error_lr = np.mean(errors_lr)

    # Création d'un DataFrame pour afficher y_pred_lr et y_train côte à côte
    comparison_df = pd.DataFrame({'y_test': y_test, 'y_pred_lr': y_pred_lr})
    # Affichage des valeurs
    #print("RESULTATS PREDICTIONS AVEC 1 ESSAI")
    #print(comparison_df)

    # Affichage de la moyenne et des écarts individuels
    #print(f"Error : {mean_error_lr}")
    #print(f"Errors (Linear Regression):\n{errors_lr}")
    error.append(mean_error_lr)

print("RESULT1: ", np.mean(error))



#TEST AVEC LES 2 PREMIERS ESSAIS

# Sélectionner les données des essais 1 et 2
data_essai2 = data[data['Essai'] == 2]
# Renommer les colonnes de l'essai 2 pour éviter les doublons lors de la fusion
data_essai2 = data_essai2.rename(columns=lambda x: f"{x}_essai2" if x not in ['ID', 'Essai'] else x)
# Fusionner les données des essais 1 et 2 par ID
data_combined = pd.merge(data_grouped, data_essai2, on='ID', suffixes=('', '_2'))

# Afficher les premières lignes pour vérification
#print(data_combined.head())

# Sélection des caractéristiques pertinentes42
features = data_combined.drop(columns=['ID', 'Proposition_Phase', 'Essai', 'Essai_2', 'Proposition_Phase_essai2'])
target = essais_necessaires_df['Nombre_Essais']

# Normalisation des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

error_2=[]
for i in range(60):
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=i)

    # Modèle de régression linéaire
    lr = SVR()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Dummy Regressor
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    # Calcul des écarts entre les prédictions et les valeurs réelles
    errors_lr = abs(y_test - y_pred_lr)

    # Calcul de la moyenne des écarts
    mean_error_lr = np.mean(errors_lr)

    # Création d'un DataFrame pour afficher y_pred_lr et y_train côte à côte
    comparison_df = pd.DataFrame({'y_test': y_test, 'y_pred_lr': y_pred_lr})
    # Affichage des valeurs
    #print("RESULTATS PREDICTIONS AVEC 2 ESSAIS")
    #print(comparison_df)

    # Affichage de la moyenne et des écarts individuels
    #print(f"Error : {mean_error_lr}")
    #print(f"Errors (Linear Regression):\n{errors_lr}")
    error_2.append(mean_error_lr)

print("RESULT2: ", np.mean(error_2))


#TEST AVEC LES 3 PREMIERS ESSAIS

# Sélectionner les données des essais 1 et 2 et 3
data_essai3 = data[data['Essai'] == 3]
# Renommer les colonnes de l'essai 2 pour éviter les doublons lors de la fusion
data_essai3 = data_essai3.rename(columns=lambda x: f"{x}_essai3" if x not in ['ID', 'Essai'] else x)
# Fusionner les données des essais 1 et 2 par ID
data_combined = pd.merge(data_combined, data_essai3, on='ID', suffixes=('', '_3'))

# Afficher les premières lignes pour vérification
#print(data_combined.head())

# Sélection des caractéristiques pertinentes42
features = data_combined.drop(columns=['ID', 'Proposition_Phase', 'Essai', 'Essai_3','Proposition_Phase_essai3'])
target = essais_necessaires_df['Nombre_Essais']

# Normalisation des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

error_3=[]
for i in range(60):
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=i)

    # Modèle de régression linéaire
    lr = SVR()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Dummy Regressor
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    # Calcul des écarts entre les prédictions et les valeurs réelles
    errors_lr = abs(y_test - y_pred_lr)

    # Calcul de la moyenne des écarts
    mean_error_lr = np.mean(errors_lr)

    # Création d'un DataFrame pour afficher y_pred_lr et y_train côte à côte
    comparison_df = pd.DataFrame({'y_test': y_test, 'y_pred_lr': y_pred_lr})
    # Affichage des valeurs
    #print("RESULTATS PREDICTIONS AVEC 3 ESSAIS")
    #(comparison_df)

    # Affichage de la moyenne et des écarts individuels
    #print(f"Error : {mean_error_lr}")
    #print(f"Errors (Linear Regression):\n{errors_lr}")

    error_3.append(mean_error_lr)
print("RESULT3: ", np.mean(error_3))

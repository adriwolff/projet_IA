import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as root_mean_squared_error
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# Chargement des données
data = pd.read_csv('All_Participant_resultat_corrCollisions_delimiter_propositionFinale.csv')

print(data.describe())
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Création d'un DataFrame pour stocker le nombre d'essais nécessaires pour chaque participant pour atteindre 0 erreurs
participants = data['ID'].unique()
essais_necessaires = []

for participant in participants:
    participant_data = data[data['ID'] == participant]
    #print("ID:",participant,"essai: ", len(participant_data['Essai']))
    essais_necessaires.append({'ID': participant, 'Nombre_Essais': len(participant_data['Essai'])})


essais_necessaires_df = pd.DataFrame(essais_necessaires)
#print ("Nb d'essais nécessaires par participants:\n", essais_necessaires_df)


# Sélection des premiers essais
data_grouped = data.groupby('ID').head(1)
    
# Sélection des caractéristiques pertinentes
features = data_grouped.drop(columns=['ID', 'Proposition_Phase', 'Essai'])
target = essais_necessaires_df['Nombre_Essais']

# Normalisation des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

error=[]
error_accurate=[]
for i in range(20):
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=i)

    # Modèle de régression linéaire
    lr = SVR(C=10, epsilon=0.1, kernel='sigmoid', gamma=0.01)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    """
    # Définition de la grille de paramètres pour la recherche sur grille
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'epsilon': [0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10]
    }

    # Recherche sur grille avec validation croisée
    grid = GridSearchCV(lr, param_grid, refit=True, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Afficher les meilleurs paramètres trouvés par la recherche sur grille
    print(f"Meilleurs paramètres: {grid.best_params_}")"""

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

    for j in range(len(y_pred_lr)):
        y_pred_lr[j]=round(y_pred_lr[j])
    error_accurate.append(accuracy_score(y_test, y_pred_lr)*100)

print("RESULT1: ", np.mean(error), " / ", np.mean(error_accurate), "%")



#TEST AVEC LES 2 PREMIERS ESSAIS

# Sélectionner les données des essais 1 et 2
data_grouped = data.groupby('ID').head(2)

data_grouped.loc[:, 'Temps Experience'] = data_grouped.groupby('ID')['Temps Experience'].transform('mean')
data_grouped.loc[:, 'Temps total Inactivite'] = data_grouped.groupby('ID')['Temps total Inactivite'].transform('mean')
data_grouped.loc[:, 'Temps total Activite'] = data_grouped.groupby('ID')['Temps total Activite'].transform('mean')
data_grouped.loc[:, 'Temps total de Manipulation'] = data_grouped.groupby('ID')['Temps total de Manipulation'].transform('mean')
data_grouped.loc[:, 'Nbr Total de Manipulation'] = data_grouped.groupby('ID')['Nbr Total de Manipulation'].transform('mean')
data_grouped.loc[:, 'Temps Total Consultation Instruction'] = data_grouped.groupby('ID')['Temps Total Consultation Instruction'].transform('mean')
data_grouped.loc[:, 'Nb Total de Consultation'] = data_grouped.groupby('ID')['Nb Total de Consultation'].transform('mean')
data_grouped.loc[:, 'Nombre erreur'] = data_grouped.groupby('ID')['Nombre erreur'].transform('mean')
data_grouped = data_grouped.groupby('ID').head(1)
# Afficher les premières lignes pour vérification
#print(data_combined.head())

# Sélection des caractéristiques pertinentes42
features = data_grouped.drop(columns=['ID', 'Proposition_Phase', 'Essai'])
target = essais_necessaires_df['Nombre_Essais']

# Normalisation des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

error_2=[]
error_accurate_2=[]
for i in range(20):
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=i)

    # Modèle de régression linéaire
    lr = SVR(C=10, epsilon=0.01, kernel='rbf', gamma=0.01)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    """
    # Définition de la grille de paramètres pour la recherche sur grille
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'epsilon': [0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10]
    }

    # Recherche sur grille avec validation croisée
    grid = GridSearchCV(lr, param_grid, refit=True, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    # Afficher les meilleurs paramètres trouvés par la recherche sur grille
    print(f"Meilleurs paramètres: {grid.best_params_}")"""

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

    for j in range(len(y_pred_lr)):
        y_pred_lr[j]=round(y_pred_lr[j])

    error_accurate_2.append(accuracy_score(y_test, y_pred_lr)*100)


print("RESULT2: ", np.mean(error_2), " / ", np.mean(error_accurate_2), "%")


#TEST AVEC LES 3 PREMIERS ESSAIS

# Sélectionner les données des essais 1 et 2 et 3
data_grouped = data.groupby('ID').head(3)

data_grouped.loc[:, 'Temps Experience'] = data_grouped.groupby('ID')['Temps Experience'].transform('mean')
data_grouped.loc[:, 'Temps total Inactivite'] = data_grouped.groupby('ID')['Temps total Inactivite'].transform('mean')
data_grouped.loc[:, 'Temps total Activite'] = data_grouped.groupby('ID')['Temps total Activite'].transform('mean')
data_grouped.loc[:, 'Temps total de Manipulation'] = data_grouped.groupby('ID')['Temps total de Manipulation'].transform('mean')
data_grouped.loc[:, 'Nbr Total de Manipulation'] = data_grouped.groupby('ID')['Nbr Total de Manipulation'].transform('mean')
data_grouped.loc[:, 'Temps Total Consultation Instruction'] = data_grouped.groupby('ID')['Temps Total Consultation Instruction'].transform('mean')
data_grouped.loc[:, 'Nb Total de Consultation'] = data_grouped.groupby('ID')['Nb Total de Consultation'].transform('mean')
data_grouped.loc[:, 'Nombre erreur'] = data_grouped.groupby('ID')['Nombre erreur'].transform('mean')
data_grouped = data_grouped.groupby('ID').head(1)
# Afficher les premières lignes pour vérification
#print(data_combined.head())

# Sélection des caractéristiques pertinentes42
features = data_grouped.drop(columns=['ID', 'Proposition_Phase', 'Essai'])
target = essais_necessaires_df['Nombre_Essais']

# Normalisation des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

error_3=[]
error_accurate_3=[]
for i in range(20):
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=i)

    # Modèle de régression linéaire
    lr = SVR(C=10, epsilon=0.1, kernel='rbf', gamma=0.01)
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

    for j in range(len(y_pred_lr)):
        y_pred_lr[j]=round(y_pred_lr[j])

    error_accurate_3.append(accuracy_score(y_test, y_pred_lr)*100)

print("RESULT3: ", np.mean(error_3), " / ", np.mean(error_accurate_3),"%")

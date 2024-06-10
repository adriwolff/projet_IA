import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as root_mean_squared_error
import math

# Chargement des données
data = pd.read_csv('All_Participant_resultat_corrCollisions_delimiter_propositionFinale.csv')

# Affichage des premières lignes du dataframe pour un aperçu
print(data.head())

# Affichage des statistiques descriptives
print(data.describe())


# Création d'un DataFrame pour stocker le nombre d'essais nécessaires pour chaque participant pour atteindre 0 erreurs
participants = data['ID'].unique()
essais_necessaires = []

for participant in participants:
    participant_data = data[data['ID'] == participant]
    essais_zero_erreur = participant_data[participant_data['Nombre erreur'] == 0]
    
    if not essais_zero_erreur.empty:
        premier_essai_zero_erreur = essais_zero_erreur['Essai'].min()
        essais_necessaires.append({'ID': participant, 'Nombre_Essais': premier_essai_zero_erreur})
    else:
        essais_necessaires.append({'ID': participant, 'Nombre_Essais': participant_data['Essai'].max() + 1})

essais_necessaires_df = pd.DataFrame(essais_necessaires)
print ("Nb d'essais nécessaires par participants:\n", essais_necessaires_df)

# Fonction de prédiction
def predict_with_trials(num_trials):
    # Sélection des premiers essais
    data_grouped = data.groupby('ID').head(num_trials)
    
    # Fusionner avec les données de nombre d'essais nécessaires
    merged_data = pd.merge(data_grouped, essais_necessaires_df, on='ID')

    # Sélection des caractéristiques pertinentes
    features = merged_data.drop(columns=['ID', 'Proposition_Phase', 'Essai', 'Nombre_Essais'])
    target = merged_data['Nombre_Essais']

    # Normalisation des données
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2)

    # Modèle de régression linéaire
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Dummy Regressor
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    # Évaluation des performances
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = root_mean_squared_error(y_test, y_pred_lr)
    dummy_mae = mean_absolute_error(y_test, y_pred_dummy)
    dummy_rmse = root_mean_squared_error(y_test, y_pred_dummy)

    # Récupération des IDs correspondants
    test_indices = merged_data.index[len(X_test)]
    ids_test = []
    for i in range(test_indices):
        ids_test.append(merged_data.loc[i, 'ID'])
    print(ids_test)

    return lr_mae, lr_rmse, dummy_mae, dummy_rmse, ids_test, y_test, y_pred_lr, y_pred_dummy

def predict_time(num_trials):
    # Sélection des premiers essais
    data_grouped = data.groupby('ID').head(num_trials)
    
    # Sélection des caractéristiques pertinentes
    features = data_grouped.drop(columns=['ID', 'Proposition_Phase', 'Essai', 'Temps Experience'])
    target = data_grouped['Temps Experience']

    # Normalisation des données
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2)

    # Modèle de régression linéaire
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Dummy Regressor
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    # Évaluation des performances
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = root_mean_squared_error(y_test, y_pred_lr)
    dummy_mae = mean_absolute_error(y_test, y_pred_dummy)
    dummy_rmse = root_mean_squared_error(y_test, y_pred_dummy)

    # Récupération des IDs correspondants
    test_indices = data_grouped.index[len(X_test)]
    ids_test = data_grouped.loc[test_indices, 'ID']

    return lr_mae, lr_rmse, dummy_mae, dummy_rmse, ids_test, y_test, y_pred_lr, y_pred_dummy


# Boucle pour évaluer les performances avec 1, 2 et 3 essais
print("\n\nESTIMATION DU NOMBRE D'ESSAI POUR REUSSIR\n\n")
for num_trials in range(1, 4):
    lr_mae, lr_rmse, dummy_mae, dummy_rmse, ids_test, y_test, y_pred_lr, y_pred_dummy = predict_with_trials(num_trials)
    print(f"\nPerformance avec les {num_trials} premiers essais :")
    print(f"Linear Regression MAE: {lr_mae}")
    print(f"Linear Regression RMSE: {lr_rmse}")
    print(f"Dummy Regressor MAE: {dummy_mae}")
    print(f"Dummy Regressor RMSE: {dummy_rmse}")

    # Affichage des prédictions par ID pour le test set
    predictions_df = pd.DataFrame({
        'ID': ids_test,
        'Nombre Essais Réel': y_test,
        'Nombre Essais Prédit (LR)': y_pred_lr, 
        'Arrondi LR': [math.ceil(nombre) for nombre in y_pred_lr],
        'Nombre Essais Prédit (Dummy)': y_pred_dummy,
        #'Arrondi dummy': [math.ceil(nombre) for nombre in y_pred_dummy],
        "Ecarts entre réalité et prédiction": abs(y_test-y_pred_lr),
        "Ecart entre réalité et dummy": abs(y_test-y_pred_dummy)
    })

    print("Prédictions par ID:")
    print(predictions_df.head())
    print("\nEcart moyen entre realité et prédictions: ", abs(y_test-y_pred_lr).describe()["mean"],"\n")


print("\n\nESTIMATION DU TEMPS D'UN ESSAI\n\n")
for num_trials in range(1, 4):
    lr_mae, lr_rmse, dummy_mae, dummy_rmse, ids_test, y_test, y_pred_lr, y_pred_dummy = predict_time(num_trials)
    print(f"\nPerformance avec les {num_trials} premiers essais :")
    print(f"Linear Regression MAE: {lr_mae}")
    print(f"Linear Regression RMSE: {lr_rmse}")
    print(f"Dummy Regressor MAE: {dummy_mae}")
    print(f"Dummy Regressor RMSE: {dummy_rmse}")

    # Affichage des prédictions par ID pour le test set
    predictions_df = pd.DataFrame({
        'ID': ids_test,
        'Temps essai Réel': y_test,
        'Temps essai Prédit (LR)': y_pred_lr,
        'Temps essai Prédit (Dummy)': y_pred_dummy,
        "Ecarts entre réalité et prédiction": abs(y_test-y_pred_lr),
        "Ecart entre réalité et dummy": abs(y_test-y_pred_dummy)
    })

    print("Prédictions par ID:")
    print(predictions_df.head())
    print("\nEcart moyen entre realité et prédictions: ", abs(y_test-y_pred_lr).describe()["mean"],"\n")

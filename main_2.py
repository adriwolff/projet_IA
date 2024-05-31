import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

# Chargement des données
data = pd.read_csv('All_Participant_resultat_corrCollisions_delimiter_propositionFinale.csv')

# Fonction pour prédire le nombre d'essais nécessaires avec une régression linéaire
def predict_trials_with_cross_validation(num_trials):
    # Sélection des premiers essais
    data_grouped = data.groupby('ID').head(num_trials)
    
    # Sélection des caractéristiques pertinentes
    features = data_grouped.drop(columns=['ID', 'Proposition_Phase', 'Essai'])
    target = data_grouped['Nombre erreur']

    # Normalisation des données
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Modèle de régression linéaire
    lr = LinearRegression()

    # Prédiction avec validation croisée
    y_pred = cross_val_predict(lr, features_scaled, target, cv=5)

    return y_pred

# Calcul de l'écart moyen entre les prédictions et les valeurs réelles avec la régression linéaire
def calculate_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Calcul de l'écart moyen entre les prédictions et les valeurs réelles avec le Dummy Regressor
def calculate_dummy_mean_absolute_error(y_true):
    dummy = DummyRegressor(strategy='mean')
    dummy.fit([[0]] * len(y_true), y_true)  # Utilisation de zéros comme features factices
    y_pred_dummy = dummy.predict([[0]] * len(y_true))
    return mean_absolute_error(y_true, y_pred_dummy)

# Boucle pour évaluer les performances avec 1, 2 et 3 essais
for num_trials in range(1, 4):
    print(f"\nPerformance avec les {num_trials} premiers essais :")
    y_pred = predict_trials_with_cross_validation(num_trials)
    
    # Récupération des valeurs réelles pour chaque prédiction
    target = data.groupby('ID').head(num_trials)['Nombre erreur']

    # Calcul de l'écart moyen entre les prédictions et les valeurs réelles
    mae = calculate_mean_absolute_error(target, y_pred)
    print(f"Mean Absolute Error (Linear Regression): {mae}")

    # Calcul de l'écart moyen entre les prédictions du Dummy Regressor et les valeurs réelles
    dummy_mae = calculate_dummy_mean_absolute_error(target)
    print(f"Mean Absolute Error (Dummy Regressor): {dummy_mae}")

    # Affichage des prédictions par ID pour le nombre d'essais
    predictions_df = pd.DataFrame({
        'ID': data.groupby('ID').head(num_trials)['ID'],
        'Nombre Essais Réel': target,
        'Nombre Essais Prédit': y_pred
    })
    print("Prédictions par ID:")
    print(predictions_df.head())

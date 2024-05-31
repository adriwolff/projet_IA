import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

# Chargement des données
data = pd.read_csv('All_Participant_resultat_corrCollisions_delimiter_propositionFinale.csv')

# Fonction pour prédire le nombre d'essais nécessaires avec une régression linéaire
def predict_trials(num_trials):
    # Sélection des premiers essais
    data_grouped = data.groupby('ID').head(num_trials)
    
    # Sélection des caractéristiques pertinentes
    features = data_grouped.drop(columns=['ID', 'Essai', 'Nombre erreur', 'Proposition_Phase'])
    target = data_grouped['Nombre erreur']

    # Normalisation des données
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Modèle de régression linéaire
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # Dummy Regressor
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    return mean_absolute_error(y_test, y_pred), mean_absolute_error(y_test, y_pred_dummy), lr.predict(features_scaled)

# Simulation pour 1, 2 et 3 premiers essais
results = []
for num_trials in range(1, 4):
    mae_lr, mae_dummy, predictions = predict_trials(num_trials)
    results.append({'Nombre Essais': num_trials, 'MAE_LR': mae_lr, 'MAE_Dummy': mae_dummy, 'Predictions': predictions})

# Affichage des résultats
results_df = pd.DataFrame(results)
print(results_df)

# Affichage des prédictions par ID dans les trois cas
for index, row in results_df.iterrows():
    print(f"\nPrédictions pour chaque ID avec {row['Nombre Essais']} premiers essais:")
    print(pd.DataFrame({'ID': data['ID'].unique(), 'Nombre Essais Prédit': row['Predictions']}))


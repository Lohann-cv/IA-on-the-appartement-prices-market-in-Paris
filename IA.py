import pandas as pd
import sklearn
'''
On import l'arbre de decision par la régretion et non par la classification 
Pourquoi : quand on fait de la classification, on cherche à prédire une catégorie/classe (image = chien ou chat ?)
alors que la régression cherche à prédire une valeur numérique (prix d'une maison, température, etc.)
'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data_15 = pd.read_csv("data/data_15th.csv")
X = data_15.drop(columns=["price"])
y = data_15["price"]
# Avec la fonction si dessou on y assigne 4 set de données, 2 pour l'entrainement et 2 pour le test et leur pourcentage est deffinit par test_size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# On crée le modèle avec une profondeur de 4
# Une profondeur est un garde fou pour éviter le sur-apprentissage (overfitting=amène une précision trop aléatoire) ou le sous apprentissage (underfitting=amène une précision trop basse)
model = DecisionTreeRegressor(max_depth=4)
# On y met les données d'entrainement
model.fit(X_train, y_train)
# On fait les prédictions sur les données de test
y_pred = model.predict(X_test)
# On évalue la précision du modèle avec le score R² qui est une métrique couramment utilisée pour évaluer les modèles de régression
# Le score R² varie entre 0 et 1, où 1 indique une prédiction parfaite et 0 indique que le modèle ne fait pas mieux que la moyenne des valeurs cibles
# r2_score = r2_score(y_test, y_pred)
# print(f"Model accuracy: {r2_score:.2f}")

"""
Maintenant on va faire en sorte de pouvoir rentrer des données et d'avoir une prédiction de prix en retour
Pour ca il faut les mettres dans un dataframe avec pandas et les injecter dans le modèle pour avoir un output (notre prix)
"""


def inject_input():
    while True:
        # on crée deux variables pour y implémenter les données
        m = input("Entrez la superficie en m² : ")
        room = input("Entrez le nombre de pièces : ")
        # On en crée un tableau type exel (DataFrame) avec pandas ou on y met les valeurs inputé
        try:
            input_apartment = pd.DataFrame({"room": [int(room)],
                                            "m": [float(m)]
                                            })
            break
        except ValueError:
            print(
                "Veuillez entrer des valeurs numériques valides pour la superficie et le nombre de pièces.")
    predicted_price = model.predict(input_apartment)
    # enfin on print le résultat indication suplémentaire : .2f pour arrondir à 2 chiffres après la virgule et [0] pour avoir la valeur et pas le tableau ([0] = premier élément du tableau soit le prix trouver)
    print(
        f"Le prix prédit pour un appartement de {m} m² avec {room} pièces est de {predicted_price[0]:.2f} euros.")


inject_input()

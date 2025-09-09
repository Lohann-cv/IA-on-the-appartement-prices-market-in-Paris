import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

'''
Le modèle initial de l'IA globalement qui est dans IA_terminal.py
On utilise Streamlit qui est un module python pour faire des applications web simples et rapides
'''
data_15 = pd.read_csv("data/data_15th.csv")
X = data_15.drop(columns=["price"])
y = data_15["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(max_depth=4)
model.fit(X_train, y_train)

'''
On crée l'application web avec Streamlit
Le titre les zone d'input excetera
'''
st.title("Estimation du prix d'un appartement")

# 9m² est la superficie minimum d'un appartement à Paris (legal)
superficie = st.number_input("Superficie (m²)", min_value=9)
pieces = st.number_input("Nombre de pièces", min_value=1)

if st.button("Calculer le prix"):
    input_apartment = pd.DataFrame({"room": [pieces], "m": [superficie]})
    prix = model.predict(input_apartment)
    st.success(f"Prix estimé : {prix[0]:.2f} €")

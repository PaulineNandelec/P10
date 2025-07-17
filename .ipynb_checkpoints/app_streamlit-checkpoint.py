import streamlit as st
import pandas as pd
import joblib

st.title("Prédiction de l'authenticité des billets")

fichier = st.file_uploader("Charger un fichier CSV", type=["csv"])
if fichier is not None:
    df = pd.read_csv(fichier, index_col='id')
    # charger modèle
    modele = joblib.load("prediction_billet.pkl")
    predictions = modele.predict(df)
    proba_is_genuine = modele.predict_proba(df)[:, 1] # on ne veut que le probabilité que le billet soit vrai

    df['is_genuine'] = predictions
    df['proba_is_genuine'] = proba_is_genuine
    st.write(df)
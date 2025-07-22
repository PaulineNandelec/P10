import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Authenticité des billets",
    layout="wide",  # On garde wide pour avoir un tableau bien large
    page_icon="💵"
)

# CSS personnalisé
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-size: 18px;
            background-color: #001520;
            color: white;
        }

        .stApp {
            padding-left: 5%;
            padding-right: 5%;
        }

        h1, h2, h3, h4 {
            color: #6abee8;
        }

        .stDataFrame div[data-testid="stVerticalBlock"] {
            overflow-x: auto;
            font-size: 18px;
        }

        .stButton>button, .stFileUploader>div>div {
            background-color: #6abee8;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💵 Prédiction de l'authenticité des billets")
st.markdown("Ce modèle prédit si un billet est **authentique** ou **faux** en fonction de ses caractéristiques physiques.")

# Colonnes attendues
colonnes_attendues = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

def verifier_colonnes(df, colonnes):
    colonnes_manquantes = [col for col in colonnes if col not in df.columns]
    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes dans le fichier : {colonnes_manquantes}")

def verifier_valeurs_nulles(df, colonnes):
    colonnes_nulles = df[colonnes].columns[df[colonnes].isnull().any()].tolist()
    if colonnes_nulles:
        raise ValueError(f"Présence de valeurs nulles dans les colonnes : {colonnes_nulles}")

with st.sidebar:
    st.header("📌 Instructions")
    st.write("""
    - Chargez un fichier CSV contenant les colonnes suivantes :
      `diagonal`, `height_left`, `height_right`, `margin_low`, `margin_up`, `length`
    - Le fichier doit également contenir une colonne `id` (index).
    - Aucune valeur manquante ne doit être présente.
    """)

fichier = st.file_uploader("📎 Charger un fichier CSV", type=["csv"])

if fichier is not None:
    try:
        df = pd.read_csv(fichier)

        if 'id' not in df.columns:
            raise ValueError("La colonne 'id' est manquante dans le fichier CSV.")

        df = df.set_index('id')

        verifier_colonnes(df, colonnes_attendues)
        verifier_valeurs_nulles(df, colonnes_attendues)

        modele = joblib.load("prediction_billet.pkl")

        predictions = modele.predict(df)
        proba_is_genuine = modele.predict_proba(df)[:, 1]

        df['is_genuine'] = predictions
        df['proba_is_genuine'] = proba_is_genuine

        st.success("✅ Prédiction réalisée avec succès !")

        st.subheader("📊 Résultats des prédictions")

        st.dataframe(
            df.style.format({
                'proba_is_genuine': "{:.2%}",
                'is_genuine': lambda x: "Vrai" if x == 1 else "Faux"
            }),
            height=600,
            use_container_width=True
        )

    except ValueError as ve:
        st.error(f"❌ Erreur de validation : {ve}")

    except FileNotFoundError:
        st.error("❌ Le fichier du modèle 'prediction_billet.pkl' est introuvable sur le serveur.")

    except Exception as e:
        st.error(f"🚨 Une erreur inattendue est survenue : {e}")

else:
    st.info("⬆️ Veuillez charger un fichier CSV pour démarrer la prédiction.")

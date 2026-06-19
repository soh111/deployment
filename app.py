import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

#  Configuration de la page 
st.set_page_config(
    page_title="Segmentation RFM",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personnalisé 
st.markdown("""
<style>
    /* ====== IMPORT POLICE MODERNE ====== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ====== FOND GÉNÉRAL CLAIR ====== */
    .stApp {
        background: linear-gradient(135deg, #f5f3ff 0%, #eef2ff 100%);
        color: #1e1b2e;
        font-family: 'Inter', sans-serif;
    }

    /* ====== SIDEBAR ====== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6d28d9 0%, #7c3aed 100%);
        border: none;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* ====== TITRE PRINCIPAL ====== */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7c3aed, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        letter-spacing: -1px;
    }

    /* ====== CARTES STATS ====== */
    .stat-card {
        background: white;
        border: 1px solid #e9d5ff;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.08);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .stat-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 30px rgba(124, 58, 237, 0.18);
    }
    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #7c3aed;
    }
    .stat-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* ====== BOÎTE DE RÉSULTAT SEGMENT ====== */
    .segment-box {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        border: none;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        font-size: 1.7rem;
        font-weight: 800;
        color: white !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.35);
        animation: popIn 0.4s ease;
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.9); }
        to   { opacity: 1; transform: scale(1); }
    }

    /* ====== BOUTONS ====== */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(124, 58, 237, 0.45);
    }

    /* ====== CHAMPS DE SAISIE ====== */
    [data-testid="stNumberInput"] input {
        border-radius: 10px;
        border: 1.5px solid #e9d5ff;
    }

    /* ====== TITRES SECTIONS ====== */
    h3 {
        color: #4c1d95 !important;
        font-weight: 700 !important;
    }

    /* ====== CACHER LE MENU STREAMLIT ====== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Charger le modèle et les données ────────────────────────
@st.cache_resource
def load_model():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return kmeans, scaler

@st.cache_data
def load_data():
    rfm = pd.read_csv("rfm_data.csv")
    return rfm

kmeans, scaler = load_model()
rfm = load_data()

#  Noms des segments 
segment_names = {
    0: "💚 Clients Réguliers",
    1: "⚠️ Clients à Risque",
    2: "👑 Clients VIP",
    3: "🌟 Clients Premium"
}

#  Sidebar 
with st.sidebar:
    st.markdown("## 🛒 Segmentation RFM")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [" Accueil", " Prédiction", " Dashboard", " À propos"],
        label_visibility="hidden"
    )
    st.markdown("---")
    st.markdown("### 📈 Aperçu")
    st.metric("Clients analysés", len(rfm))
    st.metric("Segments", 4)
    
    
    # ─── PAGE ACCUEIL ─────
if page == " Accueil":
    st.markdown('<p class="main-title">🛒 Segmentation Clients RFM</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Objectif")
        st.markdown("""
        Cette application segmente les clients d'une entreprise 
        e-commerce selon leur **comportement d'achat**, grâce à 
        la méthode **RFM** et au clustering **K-Means**.
        """)

        st.markdown("### 📊 La méthode RFM")
        st.markdown("""
        - **R** (Recency) : temps depuis le dernier achat
        - **F** (Frequency) : nombre d'achats
        - **M** (Monetary) : montant total dépensé
        """)

    

        st.markdown("### 📋 Comment utiliser")
        st.markdown("""
        1. Allez dans **🔮 Prédiction**
        2. Saisissez les valeurs R, F, M d'un client
        3. Découvrez son segment et son profil
        """)

    st.markdown("---")

    # Cartes des 4 segments
    st.markdown("### 🗂️ Les 4 segments")
    col1, col2, col3, col4 = st.columns(4)

    segments_info = [
        ("💚", "Réguliers", "Clients ordinaires actifs"),
        ("⚠️", "À Risque", "Clients qui s'éloignent"),
        ("👑", "VIP", "Grands comptes précieux"),
        ("🌟", "Premium", "Meilleurs clients fidèles")
    ]

    for col, (emoji, nom, desc) in zip([col1, col2, col3, col4], segments_info):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{emoji}</div>
                <div class="stat-label"><b>{nom}</b><br>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
            
            # ─── PAGE PRÉDICTION ─────────────────────────────────────────
if page == " Prédiction":
    st.markdown('<p class="main-title"> Prédire le Segment d\'un Client</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###  Saisir les valeurs du client")

        recency = st.number_input(
            "Recency (jours depuis le dernier achat)",
            min_value=0, max_value=500, value=30
        )

        frequency = st.number_input(
            "Frequency (nombre d'achats)",
            min_value=1, max_value=300, value=5
        )

        monetary = st.number_input(
            "Monetary (montant total dépensé en €)",
            min_value=0.0, max_value=300000.0, value=1000.0
        )

        predict_btn = st.button("🔍 Trouver le segment")

    with col2:
        st.markdown("### 📊 Résultat")

        if predict_btn:
            # Préparer les données
            client_data = np.array([[recency, frequency, monetary]])
            client_scaled = scaler.transform(client_data)

            # Prédiction
            cluster = kmeans.predict(client_scaled)[0]
            segment = segment_names[cluster]

            # Afficher le segment
            st.markdown(f"""
            <div class="segment-box">
                {segment}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Profil détaillé selon le cluster
            profils = {
                0: "Client actif au comportement standard. Achète régulièrement avec un budget modéré.",
                1: "Client en perte de vitesse. N'a pas acheté depuis longtemps — à réactiver avec des offres ciblées.",
                2: "Client exceptionnel à très forte valeur. Probablement un grossiste ou grand compte. À choyer en priorité.",
                3: "Excellent client fidèle. Achète souvent et dépense bien — un pilier de l'entreprise."
            }

            st.info(profils[cluster])

        else:
            st.info(" Saisissez les valeurs et cliquez sur le bouton")
            
            
    
 # ─── PAGES EN CONSTRUCTION (temporaire) ──────────────────────
# ─── PAGE DASHBOARD ──────────────────────────────────────────
if page == " Dashboard":
    st.markdown('<p class="main-title">📊 Tableau de Bord des Segments</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Calcul des stats par cluster ──
    cluster_counts = rfm['Cluster'].value_counts().sort_index()
    cluster_profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

    # ── Cartes : nombre de clients par segment ──
    st.markdown("### 🗂️ Répartition des clients")
    col1, col2, col3, col4 = st.columns(4)

    cols = [col1, col2, col3, col4]
    for i in range(4):
        with cols[i]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{cluster_counts[i]}</div>
                <div class="stat-label">{segment_names[i]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Deux graphiques côte à côte ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###  Répartition des segments")

        fig, ax = plt.subplots(figsize=(5, 5))
        colors = ['#22c55e', '#ef4444', '#f59e0b', '#7c3aed']
        labels = ['Réguliers', 'À Risque', 'VIP', 'Premium']

        ax.pie(
    cluster_counts,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.85,
    labeldistance=1.1
)
        st.pyplot(fig)

    with col2:
        st.markdown("### 📊 Profil moyen par segment")

        # Normaliser pour comparaison
        profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())

        fig2, ax2 = plt.subplots(figsize=(5, 5))
        x = np.arange(4)
        width = 0.25

        ax2.bar(x - width, profile_norm['Recency'], width, label='Recency', color='#ef4444')
        ax2.bar(x, profile_norm['Frequency'], width, label='Frequency', color='#7c3aed')
        ax2.bar(x + width, profile_norm['Monetary'], width, label='Monetary', color='#22c55e')

        ax2.set_xticks(x)
        ax2.set_xticklabels(['C0', 'C1', 'C2', 'C3'])
        ax2.legend()
        st.pyplot(fig2)

    st.markdown("---")

    # ── Tableau récapitulatif ──
    st.markdown("### 📋 Profil détaillé des segments")

    recap = cluster_profile.copy()
    recap['Nb_clients'] = cluster_counts
    recap['Segment'] = [segment_names[i] for i in range(4)]
    recap = recap.round(1)
    recap = recap[['Segment', 'Recency', 'Frequency', 'Monetary', 'Nb_clients']]

    st.dataframe(recap, use_container_width=True)
# ─── PAGE À PROPOS ───────────────────────────────────────────
if page == " À propos":
    st.markdown('<p class="main-title">ℹ️ À propos du projet</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Le projet")
        st.markdown("""
        Cette application a été développée dans le cadre du cours 
        **Framework Machine Learning — Apprentissage non supervisé**.

        Elle segmente automatiquement les clients d'une entreprise 
        e-commerce selon leur comportement d'achat, afin d'aider 
        les équipes marketing à cibler leurs actions.
        """)

        st.markdown("### 🔬 La méthode")
        st.markdown("""
        1. **Nettoyage** des données (valeurs nulles, doublons)
        2. **Feature Engineering** : calcul des variables RFM
        3. **Normalisation** avec StandardScaler
        4. **Clustering** avec K-Means (K=4)
        5. **Validation** : méthode du coude + score de silhouette
        """)

    with col2:
        st.markdown("### 🛠️ Technologies")
        st.markdown("""
        - **Python** — langage principal
        - **Pandas** — manipulation des données
        - **Scikit-learn** — K-Means et normalisation
        - **Matplotlib** — visualisations
        - **Streamlit** — interface web
        """)

        st.markdown("### 📊 Le dataset")
        st.markdown("""
        - **Source** : Online Retail (e-commerce UK)
        - **Volume initial** : 541 909 transactions
        - **Après nettoyage** : 401 604 transactions
        - **Clients uniques** : 4 338
        - **Période** : 2010 - 2011
        """)

    st.markdown("---")

    # Signature
    st.markdown("""
    <div style="text-align:center; color:#6b7280; padding:1rem;">
        Développé par <b>Soh Honoré Laurent</b> — KEYCE Informatique<br>
        IABD B3 — 2025/2026
    </div>
    """, unsafe_allow_html=True)

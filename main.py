import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('df_reduit.csv')

# Encode categorical variables
encoding_dict = {
    'depenses_recurrentes': {'Oui': 1, 'Non': 0},
    'kit_concurrence': {'Oui': 1, 'Non': 0},
    'cout_kit_credit': {'Oui': 1, 'Non': 0},
    'duree_pret_comprise': {'Oui': 1, 'Non': 0},
    'traite_comprise': {'Oui': 1, 'Non': 0},
    'montant_correspondant': {'Oui': 1, 'Non': 0},
    'accord_penalites_retard': {'Oui': 1, 'Non': 0},
    'informer_code_ussd': {'Oui': 1, 'Non': 0},
    'numero_cx_communique': {'Oui': 1, 'Non': 0},
    'informer_5j_arrieres_45j': {'Oui': 1, 'Non': 0},
    'informer_90j_retard': {'Oui': 1, 'Non': 0},
    'Shop': {
        'Dagana': 0, 'Dahara': 1, 'Diaroume': 2, 'Diouloulou': 3, 'Diourbel': 4,
        'Gandiaye': 5, 'Kaffrine': 6, 'Kedougou': 7, 'Kolda': 8, 'Kounghel': 9,
        'Louga': 10, 'Ndioum': 11, 'Ngaye Mekhe': 12, 'Nioro': 13, 'Saint Louis': 14,
        'Tambacounda': 15, 'Velingara': 16, 'Ziguinchor': 17
    },
    'product_name': {'Bpower60-Kit-TV': 0, 'bPower60 WO TV-Radio-Torch + FAN': 1},
    'revenu_range': {
        '50 000  et 100 000FCFA': 0, '100 000FCFA  et 200 000': 1,
        '200 000 et 300 000 FCFA': 2, '+ de 300 000 FCFA': 3
    },
    'motivation_kit': {
        'UTILISATION GENERAL (Tv, Lumiere, ventilateur, recharge telephone)': 0,
        'LUMIERE ET TV': 1, 'BUSINESS': 2, 'ZONE NON ELECTRIFIE': 3, 'QUALITE DU KIT': 4,
        'A CAUSE DES DELESTAGES': 5, 'LE PRIX EST ABORDABLE': 6, 'CHARGER SON TELEPHONE': 7,
        'POUR LA FAMILLE': 8, 'LA TELE': 9
    },
    'current_contract_status': {'repo': 0, 'active': 1},
    'metier': {'SAISONNIER': 0, 'REGULIER': 1},
    'activite_professionnelle': {
        'AGRICULTEUR': 0, 'COMMERCANT': 1, 'MECANICIEN': 2, 'BOUCHER': 3, 'TAILLEUR': 4,
        'ENSEIGNANT': 5, 'ELEVEUR': 6, 'PECHEUR': 7, 'BUSINESS': 8, 'BOULANGER': 9,
        'COUTURIER': 10, 'CULTIVATEUR': 11, 'ETUDIANT': 12, 'GENDARME': 13, 'PEINTRE': 14
    }
}

# Decode dictionary to map from encoded values back to original strings
decode_dict = {feature: {v: k for k, v in mapping.items()} for feature, mapping in encoding_dict.items()}

for column, mapping in encoding_dict.items():
    df[column] = df[column].replace(mapping)

# Feature extraction
X = df[['product_name', 'Shop', 'revenu_range', 'depenses_mensuelles', 'depenses_recurrentes', 'motivation_kit',
        'kit_concurrence', 'cout_kit_credit', 'duree_pret_comprise', 'traite_comprise', 'montant_correspondant',
        'accord_penalites_retard', 'informer_code_ussd', 'numero_cx_communique', 'informer_5j_arrieres_45j',
        'informer_90j_retard', 'metier', 'activite_professionnelle']]
y = df['current_contract_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Streamlit app
st.markdown('''
<center>
<h1>Screening App - Évaluation des Clients pour Kits Solaires PAYG</h1>
</center>
''', unsafe_allow_html=True)

st.header('''
Bienvenue dans l'application de screening de notre entreprise spécialisée dans l'énergie renouvelable. Cette application a pour but d'évaluer les candidats souhaitant acquérir un kit solaire en mode Pay-As-You-Go (PAYG). En répondant à ce questionnaire, nous pourrons déterminer si vous êtes éligible pour un contrat de financement, permettant ainsi de bénéficier d'un kit solaire à crédit.
''')

from PIL import Image
img = Image.open("Bboxx_Africa.jpg")
st.image(img, width=750)

st.sidebar.header("Questionnaire")


# Define a function to get user input
def user_input_features():
    questions = {
        "Combien dépensez vous par mois (dépense quotidienne et ration alimentaire)? *Selectionner un montant proche de vos dépenses": "depenses_mensuelles",
        "Avez vous d'autres dépenses récurrentes?": "depenses_recurrentes",
        "Qu'est-ce qui vous motive à prendre le kit": "motivation_kit",
        "Est ce que le client a un Kit qu'il paye déja chez la concurrence ?": "kit_concurrence",
        "Est ce que le client comprend combien lui coute le kit à crédit ?": "cout_kit_credit",
        "Est ce que le client comprend la durée du prêt ?": "duree_pret_comprise",
        "Est-ce que le client comprend combien il doit payer pour chaque traite ?": "traite_comprise",
        "Est-ce que le montant communiqué par l'agent commercial correspond à celui indiqué par le client ?": "montant_correspondant",
        "Est ce que le client est d'accord sur le principe des pénalités liées au paiement en retard ?": "accord_penalites_retard",
        "Informer le client sur le code USSD et sur le fait qu'il doit lui meme effectuer ses paiements (le cas echeant PEG dégage ses responsabilités)": "informer_code_ussd",
        "Est-ce le numéro du CX est communiquer au client ? ": "numero_cx_communique",
        "Informer le client sur le fait qu'il ne doit pas accumuler 5 jours d'arriérés durant les 45 premiers jours sinon son compte sera suspendu.": "informer_5j_arrieres_45j",
        "Informer le client sur le fait qu'il ne doit pas cumuler 90 jours de retard, siNo le kit sera récupéré.": "informer_90j_retard",
        "Metier": "metier",
        "Quelle est votre activité professionnelle ?": "activite_professionnelle",
        "Quelle est votre tranche de revenu ?": "revenu_range",
        "Ville": "Shop",
        "Produit choisi": "product_name"
    }

    input_data = {}
    for question, feature in questions.items():
        if feature in encoding_dict:
            options = list(decode_dict[feature].values())
            response = st.sidebar.selectbox(question, options)
            input_data[feature] = encoding_dict[feature][response]
        else:
            if feature == 'depenses_mensuelles':
                options = [150000, 200000, 250000, 400000, 500000, 600000]
                response = st.sidebar.selectbox(question, options)
                input_data[feature] = response
            else:
                input_data[feature] = st.sidebar.number_input(question)

    return pd.DataFrame([input_data])


# Get user input
input_df = user_input_features()

# Ensure the input_df columns match the order and names used in training
input_df = input_df[X.columns]

# Predict the contract status
prediction = log_reg.predict(input_df)

# Display the prediction
st.subheader('Prediction')
st.write(f"{'Le client pourrait bénéficier du kit solaire. Et effectuer un paiement régulier.' if prediction[0] == 1 else 'Client pourrait ne pas payer régulièrement. Et voir son kit solaire reposséder'}")

if st.sidebar.button("Validate"):
    # Predict and display the result
    prediction = log_reg.predict(input_df)
    st.subheader(f"Client Status Prediction: {'Active' if prediction[0] == 1 else 'Repossession'}")
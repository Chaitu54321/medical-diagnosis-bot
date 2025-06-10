import joblib
import pandas as pd
import difflib
import streamlit as st

# Load model
model = joblib.load("model/disease_prediction_model.pkl")

# Load additional data
desc_df = pd.read_csv("dispred dataset/symptom_description.csv")
precaution_df = pd.read_csv("dispred dataset/symptom_precaution.csv")
severity_df = pd.read_csv("dispred dataset/symptom_severity.csv")

# Prepare dictionaries for fast lookup
desc_dict = dict(zip(desc_df['Disease'], desc_df['Description']))
precaution_dict = precaution_df.set_index('Disease').T.to_dict('list')
severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))


# Load the explicit list of symptoms representing training feature order
TRAINING_SYMPTOM_ORDER = [] # Initialize to ensure it exists
try:
    TRAINING_SYMPTOM_ORDER = joblib.load("model/symptom_list.pkl")
    if not isinstance(TRAINING_SYMPTOM_ORDER, list): # Ensure it's a list
        print(f"Warning: model/symptom_list.pkl did not load as a list (type: {type(TRAINING_SYMPTOM_ORDER)}). Attempting conversion.")
        TRAINING_SYMPTOM_ORDER = list(TRAINING_SYMPTOM_ORDER)
    print(f"Successfully loaded TRAINING_SYMPTOM_ORDER from model/symptom_list.pkl (count: {len(TRAINING_SYMPTOM_ORDER)})")
except Exception as e_pkl_load:
    print(f"ERROR: Could not load model/symptom_list.pkl. Error: {e_pkl_load}")
    print("Warning: Falling back to model.feature_names_in_ for TRAINING_SYMPTOM_ORDER.")
    # Fallback to model.feature_names_in_ if pkl fails
    if 'model' in locals() and hasattr(model, 'feature_names_in_'):
        try:
            TRAINING_SYMPTOM_ORDER = list(model.feature_names_in_) # Ensure it's a list
            print(f"Successfully used model.feature_names_in_ (count: {len(TRAINING_SYMPTOM_ORDER)})")
        except Exception as e_feat_names:
            print(f"ERROR: Could not get feature_names_in_ from model. Error: {e_feat_names}")
            TRAINING_SYMPTOM_ORDER = [] # Critical failure
    else:
        print("CRITICAL ERROR: Model not available for feature_names_in_ fallback for TRAINING_SYMPTOM_ORDER.")
        TRAINING_SYMPTOM_ORDER = []

if not TRAINING_SYMPTOM_ORDER:
    print("CRITICAL ERROR: TRAINING_SYMPTOM_ORDER is empty. Prediction input vector will be incorrect.")
    # Define a minimal dummy list if it's empty to prevent downstream syntax errors, though predictions will be wrong
    # This case should indicate a severe setup problem.
    TRAINING_SYMPTOM_ORDER = ['dummy_symptom_fallback']


# Explicitly use the loaded list for consistency in the global 'all_symptoms' variable.
# This ensures map_user_input and input_vector construction use the definitive ordered list.
all_symptoms = TRAINING_SYMPTOM_ORDER



# List of all symptoms used in training
# all_symptoms originally model.feature_names_in_, now loaded from TRAINING_SYMPTOM_ORDER
phrase_to_symptom = {
    "abdominal pain": "abdominal_pain",
    "abnormal menstruation": "abnormal_menstruation",
    "aching joints": "joint_pain",
    "acidity": "acidity",
    "acute liver failure": "acute_liver_failure",
    "alcohol history": "history_of_alcohol_consumption",
    "altered sensorium": "altered_sensorium",
    "anal pain": "pain_in_anal_region",
    "anxiety": "anxiety",
    "back ache": "back_pain",
    "back pain": "back_pain",
    "balance loss": "loss_of_balance",
    "being sick": "vomiting",
    "belly cramps": "abdominal_pain",
    "belly pain": "belly_pain",
    "blackheads": "blackheads",
    "bladder discomfort": "bladder_discomfort",
    "bleeding stomach": "stomach_bleeding",
    "blister": "blister",
    "blisters": "blister",
    "blood in sputum": "blood_in_sputum",
    "blood while coughing": "blood_in_sputum",
    "bloody stool": "bloody_stool",
    "blurred and distorted vision": "blurred_and_distorted_vision",
    "blurred vision": "blurred_and_distorted_vision",
    "body pain": "malaise",
    "body side weakness": "weakness_of_one_body_side",
    "breathing difficulty": "breathlessness",
    "breathlessness": "breathlessness",
    "brittle nails": "brittle_nails",
    "bruising": "bruising",
    "burning micturition": "burning_micturition",
    "burning up": "high_fever",
    "can't breathe properly": "breathlessness",
    "can't focus": "lack_of_concentration",
    "can't smell": "loss_of_smell",
    "chest pain": "chest_pain",
    "chills": "chills",
    "cold": "continuous_sneezing",
    "cold hands": "cold_hands_and_feets",
    "cold hands and feets": "cold_hands_and_feets",
    "coma": "coma",
    "confused state": "altered_sensorium",
    "congestion": "congestion",
    "constipation": "constipation",
    "continuous feel of urine": "continuous_feel_of_urine",
    "continuous sneezing": "continuous_sneezing",
    "cough": "cough",
    "cramps": "cramps",
    "dark pee": "dark_urine",
    "dark urine": "dark_urine",
    "dehydration": "dehydration",
    "depression": "depression",
    "diarrhoea": "diarrhoea",
    "difficulty breathing": "breathlessness",
    "dischromic  patches": "dischromic__patches",
    "distention of abdomen": "distention_of_abdomen",
    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "dry lips": "drying_and_tingling_lips",
    "drying and tingling lips": "drying_and_tingling_lips",
    "enlarged thyroid": "enlarged_thyroid",
    "excessive hunger": "excessive_hunger",
    "excessive sweating": "sweating",
    "exhausted": "fatigue",
    "extra marital contacts": "extra_marital_contacts",
    "extreme tiredness": "fatigue",
    "eye pain": "pain_behind_the_eyes",
    "eye problems": "visual_disturbances",
    "eye redness": "redness_of_eyes",
    "eye swelling": "puffy_face_and_eyes",
    "family history": "family_history",
    "fast heart rate": "fast_heart_rate",
    "fast heartbeat": "fast_heart_rate",
    "fatigue": "fatigue",
    "feel like vomiting": "nausea",
    "feeling anxious": "anxiety",
    "feeling cold": "chills",
    "feeling shivery": "chills",
    "feeling sick": "nausea",
    "feeling thirsty": "excessive_hunger",
    "fever": "high_fever",
    "fluid overload": "fluid_overload",
    "foul smell of urine": "foul_smell_of_urine",
    "frequent pee": "continuous_feel_of_urine",
    "frequent sneezing": "continuous_sneezing",
    "frequent urination": "continuous_feel_of_urine",
    "gas": "passage_of_gases",
    "head ache": "headache",
    "head pain": "headache",
    "headache": "headache",
    "heartburn": "acidity",
    "high fever": "high_fever",
    "high temperature": "high_fever",
    "hip joint pain": "hip_joint_pain",
    "hip pain": "hip_joint_pain",
    "history of alcohol consumption": "history_of_alcohol_consumption",
    "hungry often": "excessive_hunger",
    "increased appetite": "increased_appetite",
    "indigestion": "indigestion",
    "inflammatory nails": "inflammatory_nails",
    "internal itching": "internal_itching",
    "irregular sugar level": "irregular_sugar_level",
    "irritability": "irritability",
    "irritation": "irritability",
    "irritation in anus": "irritation_in_anus",
    "itching": "itching",
    "joint pain": "joint_pain",
    "joint swelling": "swelling_joints",
    "knee pain": "knee_pain",
    "lack of concentration": "lack_of_concentration",
    "leg cramps": "cramps",
    "lethargy": "lethargy",
    "limb weakness": "weakness_in_limbs",
    "liver failure": "acute_liver_failure",
    "loss of appetite": "loss_of_appetite",
    "loss of balance": "loss_of_balance",
    "loss of smell": "loss_of_smell",
    "loss of taste": "loss_of_taste",
    "low appetite": "loss_of_appetite",
    "low energy": "lethargy",
    "low fever": "mild_fever",
    "malaise": "malaise",
    "mild fever": "mild_fever",
    "mood issues": "mood_swings",
    "mood swings": "mood_swings",
    "movement stiffness": "movement_stiffness",
    "mucoid sputum": "mucoid_sputum",
    "muscle loss": "muscle_wasting",
    "muscle pain": "muscle_pain",
    "muscle wasting": "muscle_wasting",
    "muscle weakness": "muscle_weakness",
    "nausea": "nausea",
    "neck pain": "neck_pain",
    "neck stiffness": "stiff_neck",
    "no appetite": "loss_of_appetite",
    "nodal skin eruptions": "nodal_skin_eruptions",
    "non-stop sneezing": "continuous_sneezing",
    "nose blocked": "congestion",
    "not hungry": "loss_of_appetite",
    "obesity": "obesity",
    "pain behind the eyes": "pain_behind_the_eyes",
    "pain during bowel movements": "pain_during_bowel_movements",
    "pain in anal region": "pain_in_anal_region",
    "pain in head": "headache",
    "painful walking": "painful_walking",
    "palpitations": "palpitations",
    "passage of gases": "passage_of_gases",
    "patches in throat": "patches_in_throat",
    "period issues": "abnormal_menstruation",
    "phlegm": "phlegm",
    "pimples": "pus_filled_pimples",
    "pimples on face": "pus_filled_pimples",
    "polyuria": "polyuria",
    "prominent veins on calf": "prominent_veins_on_calf",
    "puffy face and eyes": "puffy_face_and_eyes",
    "puking": "vomiting",
    "pus filled pimples": "pus_filled_pimples",
    "rash on skin": "skin_rash",
    "rashes": "skin_rash",
    "receiving blood transfusion": "receiving_blood_transfusion",
    "receiving unsterile injections": "receiving_unsterile_injections",
    "red dots": "red_spots_over_body",
    "red nose sores": "red_sore_around_nose",
    "red sore around nose": "red_sore_around_nose",
    "red spots over body": "red_spots_over_body",
    "redness of eyes": "redness_of_eyes",
    "restless": "restlessness",
    "restlessness": "restlessness",
    "runny nose": "runny_nose",
    "rusty sputum": "rusty_sputum",
    "sadness": "depression",
    "scurring": "scurring",
    "shivering": "shivering",
    "shortness of breath": "breathlessness",
    "silver like dusting": "silver_like_dusting",
    "sinus pain": "sinus_pressure",
    "sinus pressure": "sinus_pressure",
    "skin bruises": "bruising",
    "skin irritation": "skin_rash",
    "skin peeling": "skin_peeling",
    "skin rash": "skin_rash",
    "slurred speech": "slurred_speech",
    "slurred words": "slurred_speech",
    "small dents in nails": "small_dents_in_nails",
    "sneezing": "continuous_sneezing",
    "sneezing a lot": "continuous_sneezing",
    "sore head": "headache",
    "sore joints": "joint_pain",
    "sore throat": "throat_irritation",
    "spinning movements": "spinning_movements",
    "spotting  urination": "spotting__urination",
    "stiff movements": "movement_stiffness",
    "stiff neck": "stiff_neck",
    "stomach ache": "abdominal_pain",
    "stomach bleeding": "stomach_bleeding",
    "stomach cramps": "abdominal_pain",
    "stomach pain": "stomach_pain",
    "sunken eyes": "sunken_eyes",
    "sweating": "sweating",
    "sweating a lot": "sweating",
    "swelled lymph nodes": "swelled_lymph_nodes",
    "swelling joints": "swelling_joints",
    "swelling of stomach": "swelling_of_stomach",
    "swollen blood vessels": "swollen_blood_vessels",
    "swollen extremeties": "swollen_extremeties",
    "swollen legs": "swollen_legs",
    "swollen neck": "enlarged_thyroid",
    "swollen neck nodes": "swelled_lymph_nodes",
    "swollen stomach": "swelling_of_stomach",
    "throat irritation": "throat_irritation",
    "throat patches": "patches_in_throat",
    "throwing up": "vomiting",
    "tired": "fatigue",
    "toxic look (typhos)": "toxic_look_(typhos)",
    "tummy ache": "abdominal_pain",
    "ulcers on tongue": "ulcers_on_tongue",
    "unsteadiness": "unsteadiness",
    "upset stomach": "nausea",
    "urine a lot": "polyuria",
    "urine burn": "burning_micturition",
    "urine pain": "burning_micturition",
    "urine smell": "foul_smell_of urine",
    "very tired": "fatigue",
    "visual disturbances": "visual_disturbances",
    "vomiting": "vomiting",
    "walking pain": "painful_walking",
    "watering from eyes": "watering_from_eyes",
    "watery eyes": "watering_from_eyes",
    "weak nails": "brittle_nails",
    "weakness in limbs": "weakness_in_limbs",
    "weakness of one body side": "weakness_of_one_body_side",
    "weight gain": "weight_gain",
    "weight loss": "weight_loss",
    "worn out": "fatigue",
    "yellow crust ooze": "yellow_crust_ooze",
    "yellow eyes": "yellowing_of_eyes",
    "yellow skin": "yellowish_skin",
    "yellow urine": "yellow_urine",
    "yellowing of eyes": "yellowing_of_eyes",
    "yellowish skin": "yellowish_skin"
}

# --- USER INPUT ---
# user_input = input("Enter your symptoms (comma-separated): ").lower().split(",")
# user_input = [sym.strip() for sym in user_input]


def map_user_input(user_input):
    mapped_input = []
    unrecognized_phrases = [] # To inform user about phrases not understood

    # Ensure all_symptoms and phrase_to_symptom are accessible from global scope in app.py
    # These are typically loaded/defined at the beginning of app.py

    for symptom_phrase in user_input:
        symptom_phrase_processed = symptom_phrase.strip().lower()
        if not symptom_phrase_processed:
            continue

        # 1. Direct mapping using the expanded phrase_to_symptom dictionary
        if symptom_phrase_processed in phrase_to_symptom:
            mapped_symptom = phrase_to_symptom[symptom_phrase_processed]
            if mapped_symptom in all_symptoms: # Verify the mapped symptom is a recognized model feature
                mapped_input.append(mapped_symptom)
                # print(f"🤖 Interpreted '{symptom_phrase}' as '{mapped_symptom}' via direct phrase map.")
                continue # Move to next symptom_phrase
            # else:
                # This case implies phrase_to_symptom maps to a non-existent canonical symptom.
                # print(f"⚠️ Internal Warning: Phrase '{symptom_phrase_processed}' maps to '{mapped_symptom}', which is NOT in all_symptoms.")
                # Fall through to fuzzy matching against all_symptoms for the original phrase

        # 2. Fuzzy matching directly against the canonical `all_symptoms` list
        # This is useful if the user's input is close to a canonical symptom name
        direct_fuzzy_matches = difflib.get_close_matches(symptom_phrase_processed, all_symptoms, n=1, cutoff=0.8) # Adjusted cutoff
        if direct_fuzzy_matches:
            matched_symptom = direct_fuzzy_matches[0]
            mapped_input.append(matched_symptom)
            # print(f"🤖 Interpreted '{symptom_phrase}' as '{matched_symptom}' via fuzzy match on all_symptoms.")
            continue

        # 3. Fallback: Fuzzy matching against the keys of `phrase_to_symptom`
        # This helps if the user's input is close to a defined phrase key
        phrase_key_fuzzy_matches = difflib.get_close_matches(symptom_phrase_processed, phrase_to_symptom.keys(), n=1, cutoff=0.8)
        if phrase_key_fuzzy_matches:
            matched_phrase_key = phrase_key_fuzzy_matches[0]
            mapped_symptom = phrase_to_symptom[matched_phrase_key]
            if mapped_symptom in all_symptoms: # Ensure this path also leads to a valid canonical symptom
                mapped_input.append(mapped_symptom)
                # print(f"🤖 Interpreted '{symptom_phrase}' as '{matched_phrase_key}' (phrase) -> '{mapped_symptom}'.")
                continue
            # else:
                # print(f"⚠️ Internal Warning: Fuzzy phrase '{symptom_phrase_processed}' to '{matched_phrase_key}' maps to '{mapped_symptom}', not in all_symptoms.")

        # 4. If no mapping or fuzzy match, the symptom is considered unrecognized for model input
        # print(f"⚠️ Warning: Symptom '{symptom_phrase}' not recognized by any method. It will be ignored for prediction.")
        unrecognized_phrases.append(symptom_phrase) # Collect unrecognized original phrases

    # Remove duplicates while preserving order
    seen = set()
    ordered_unique_mapped_input = []
    for x in mapped_input:
        if x not in seen:
            ordered_unique_mapped_input.append(x)
            seen.add(x)

    # For Streamlit, it's good to provide feedback about unrecognized symptoms directly in the UI if possible.
    # This function, as is, doesn't directly interact with Streamlit's UI elements for warnings.
    # Consider modifying the Streamlit part of app.py to display these `unrecognized_phrases`.
    # For now, printing to console (if app is run in console) or logging.
    if unrecognized_phrases:
        print(f"⚠️ Unrecognized symptom phrases: {unrecognized_phrases}. These will be ignored.")

    return ordered_unique_mapped_input

st.set_page_config(page_title="🩺 Medical Diagnosis Bot")
st.title("🩺 Symptom Checker")
st.write("Enter your symptoms separated by commas (e.g., `headache, vomiting, tired`)")

user_symptoms = st.text_input("Your Symptoms")

if st.button("Predict Disease"):
    if user_symptoms.strip() == "":
        st.warning("Please enter at least one symptom.")
    else:
        raw_input = [s.strip().lower() for s in user_symptoms.split(",")]
        user_input = map_user_input(raw_input)

        # Create input vector (list of 0s and 1s)
        # all_symptoms should be the globally defined, correctly ordered list from symptom_list.pkl
        input_vector_list = [1 if symptom in user_input else 0 for symptom in all_symptoms]

        # Convert to Pandas DataFrame with feature names
        # This is the key change to address the UserWarning from scikit-learn
        input_df = pd.DataFrame([input_vector_list], columns=all_symptoms)

        # Predict using the DataFrame
        prediction = model.predict(input_df)[0]

        # --- BEGIN: Custom logic for handling common/mild symptoms leading to severe predictions ---

        # Canonical names for very common/mild symptoms
        VERY_COMMON_MILD_SYMPTOMS_CANONICAL = [
            'continuous_sneezing', 'runny_nose', 'throat_irritation', 'congestion',
            'loss_of_smell', 'mild_fever', 'cough', # 'headache' can be tricky
        ]

        # Keywords to identify potentially 'severe' diseases from the prediction string
        HEURISTIC_SEVERE_DISEASES_KEYWORDS = [
            'AIDS', 'Paralysis', 'Tuberculosis', 'Pneumonia', 'Heart attack',
            'Hepatitis', # Catches all Hepatitis A, B, C, D, E
            'Diabetes', # Note: Diabetes needs space if it's "Diabetes " in dataset, handled below
            'Typhoid', 'Malaria', 'Dengue', 'Jaundice'
        ]
        # Add stripped version for diabetes if dataset has "Diabetes " (desc_dict is global in app.py)
        # Also check for Hypertension
        if "Diabetes " in desc_dict:
            if "Diabetes" not in HEURISTIC_SEVERE_DISEASES_KEYWORDS: HEURISTIC_SEVERE_DISEASES_KEYWORDS.append("Diabetes ")
        if "Hypertension " in desc_dict: # Example if Hypertension was also considered severe
            if "Hypertension" not in HEURISTIC_SEVERE_DISEASES_KEYWORDS: HEURISTIC_SEVERE_DISEASES_KEYWORDS.append("Hypertension ")


        is_input_common_mild_only = False
        if user_input: # user_input is the list of mapped canonical symptoms
            is_input_common_mild_only = all(symptom in VERY_COMMON_MILD_SYMPTOMS_CANONICAL for symptom in user_input)

        is_prediction_severe = False
        # Clean the prediction string by removing potential extra spaces or variations before keyword check
        # For example, if prediction is "Diabetes  " (with trailing spaces from dataset)
        cleaned_prediction_for_check = prediction.strip().lower()
        for keyword in HEURISTIC_SEVERE_DISEASES_KEYWORDS:
            # Ensure keyword matching is case-insensitive and handles potential variations
            if keyword.strip().lower() in cleaned_prediction_for_check:
                is_prediction_severe = True
                break

        custom_message = None
        if is_input_common_mild_only and is_prediction_severe:
            custom_message = (
                f"The symptoms you provided ({', '.join(user_input)}) are common and can be associated with various conditions. "
                f"While the system identified a possibility of **{prediction}**, it might also indicate a simpler condition "
                f"(e.g., a common cold or allergy), especially if your symptoms are recent and not worsening significantly. "
                f"Please monitor your symptoms. If they persist, worsen, or if you have any concerns, it is advisable to consult a doctor for a comprehensive evaluation."
            )
            # Using st.warning to display this message prominently.
            st.warning(f"⚠️ **Important Note Regarding Your Result:** {custom_message}")
            # The original prediction will still be shown by the subsequent st.success call.

        # --- END: Custom logic ---
\n

        # Show output
        st.success(f"✅ **Predicted Disease:** {prediction}")

        # Description
        st.subheader("📄 Description")
        st.info(desc_dict.get(prediction, "No description available."))

        # Precautions
        st.subheader("🩺 Precautions")
        for precaution in precaution_dict.get(prediction, ["No precautions available."]):
            st.write(f"- {precaution}")

        # Severity
        weights = [severity_dict.get(sym, 0) for sym in user_input]
        if weights:
            avg_severity = sum(weights) / len(weights)
            st.subheader("📊 Severity Level")
            if avg_severity > 5:
                st.error("⚠️ High - Please consult a doctor immediately.")
            elif avg_severity > 3:
                st.warning("🟡 Moderate - Monitor and take precautions.")
            else:
                st.success("🟢 Low - Seems manageable.")
        else:
            st.warning("⚠️ Some symptoms may not be recognized.")
























# raw_input = input("Enter your symptoms (comma-separated): ").lower().split(",")
# user_input = map_user_input([sym.strip() for sym in raw_input])
# # --- Load the list of all symptoms ---

    

# # --- Create input vector ---
# input_vector = [1 if symptom in user_input else 0 for symptom in all_symptoms]

# # --- Predict ---
# prediction = model.predict([input_vector])
# predicted_disease = prediction[0]

# # --- Output ---
# print(f"\n✅ Predicted Disease: {predicted_disease}")

# # Description
# print("\n📄 Description:")
# print(desc_dict.get(predicted_disease, "No description available."))

# # Precautions
# print("\n🩺 Precautions:")
# for precaution in precaution_dict.get(predicted_disease, ["No precautions available."]):
#     print(f"- {precaution}")

# # Severity analysis (optional)
# weights = [severity_dict.get(sym, 0) for sym in user_input]
# if weights:
#     avg_severity = sum(weights) / len(weights)
#     print("\n📊 Severity Level:", end=" ")
#     if avg_severity > 5:
#         print("⚠️ High - Please consult a doctor immediately.")
#     elif avg_severity > 3:
#         print("🟡 Moderate - Monitor and take precautions.")
#     else:
#         print("🟢 Low - Seems manageable.")
# else:
#     print("\n⚠️ Some entered symptoms are unknown or not in the database.")

# # matched = [sym for sym in user_input if sym in all_symptoms]
# # unmatched = [sym for sym in user_input if sym not in all_symptoms]

# # print("\n✅ Matched symptoms:", matched)
# # print("❌ Unmatched symptoms:", unmatched)

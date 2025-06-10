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

# List of all symptoms used in training
all_symptoms = model.feature_names_in_
phrase_to_symptom = {
    "stomach ache": "abdominal_pain",
    "belly pain": "belly_pain",
    "period issues": "abnormal_menstruation",
    "heartburn": "acidity",
    "liver failure": "acute_liver_failure",
    "confused state": "altered_sensorium",
    "feeling anxious": "anxiety",
    "back ache": "back_pain",
    "pimples": "pus_filled_pimples",
    "urine burn": "burning_micturition",
    "blood while coughing": "blood_in_sputum",
    "bloody stool": "bloody_stool",
    "blurred vision": "blurred_and_distorted_vision",
    "breathing difficulty": "breathlessness",
    "weak nails": "brittle_nails",
    "skin bruises": "bruising",
    "chest pain": "chest_pain",
    "feeling cold": "chills",
    "cold hands": "cold_hands_and_feets",
    "nose blocked": "congestion",
    "constipation": "constipation",
    "frequent urination": "continuous_feel_of_urine",
    "cold": "continuous_sneezing",
    "sneezing": "continuous_sneezing",
    "cough": "cough",
    "leg cramps": "cramps",
    "dark pee": "dark_urine",
    "dry lips": "drying_and_tingling_lips",
    "dizzy": "dizziness",
    "swollen neck": "enlarged_thyroid",
    "hungry often": "excessive_hunger",
    "fast heartbeat": "fast_heart_rate",
    "tired": "fatigue",
    "head pain": "headache",
    "fever": "high_fever",
    "joint pain": "joint_pain",
    "hip pain": "hip_joint_pain",
    "alcohol history": "history_of_alcohol_consumption",
    "indigestion": "indigestion",
    "irritation": "irritability",
    "itching": "itching",
    "knee pain": "knee_pain",
    "can't focus": "lack_of_concentration",
    "low appetite": "loss_of_appetite",
    "balance loss": "loss_of_balance",
    "can't smell": "loss_of_smell",
    "low energy": "lethargy",
    "body pain": "malaise",
    "low fever": "mild_fever",
    "mood issues": "mood_swings",
    "stiff movements": "movement_stiffness",
    "phlegm": "mucoid_sputum",
    "muscle pain": "muscle_pain",
    "muscle loss": "muscle_wasting",
    "muscle weakness": "muscle_weakness",
    "feel like vomiting": "nausea",
    "neck pain": "neck_pain",
    "obesity": "obesity",
    "eye pain": "pain_behind_the_eyes",
    "anal pain": "pain_in_anal_region",
    "walking pain": "painful_walking",
    "gas": "passage_of_gases",
    "throat patches": "patches_in_throat",
    "urine a lot": "polyuria",
    "eye swelling": "puffy_face_and_eyes",
    "red nose sores": "red_sore_around_nose",
    "red dots": "red_spots_over_body",
    "eye redness": "redness_of_eyes",
    "restless": "restlessness",
    "runny nose": "runny_nose",
    "shivering": "shivering",
    "sinus pain": "sinus_pressure",
    "skin peeling": "skin_peeling",
    "rashes": "skin_rash",
    "slurred words": "slurred_speech",
    "neck stiffness": "stiff_neck",
    "bleeding stomach": "stomach_bleeding",
    "stomach pain": "stomach_pain",
    "sunken eyes": "sunken_eyes",
    "sweating": "sweating",
    "swollen neck nodes": "swelled_lymph_nodes",
    "joint swelling": "swelling_joints",
    "swollen stomach": "swelling_of_stomach",
    "yellow skin": "yellowish_skin",
    "yellow eyes": "yellowing_of_eyes",
    "yellow urine": "yellow_urine",
    "weight loss": "weight_loss",
    "weight gain": "weight_gain",
    "vomiting": "vomiting",
    "watery eyes": "watering_from_eyes",
    "limb weakness": "weakness_in_limbs",
    "body side weakness": "weakness_of_one_body_side",
    "loss of taste": "loss_of_taste",
    "sore throat": "throat_irritation",
    "throat irritation": "throat_irritation",
    "frequent pee": "continuous_feel_of_urine",
    "urine smell": "foul_smell_of urine",
    "urine pain": "burning_micturition",
    "depression": "depression",
    "sadness": "depression",
    "anxiety": "anxiety",
    "eye problems": "visual_disturbances",
    "feeling thirsty": "excessive_hunger",
    "blisters": "blister",
    "pimples on face": "pus_filled_pimples"
}

# --- USER INPUT ---
# user_input = input("Enter your symptoms (comma-separated): ").lower().split(",")
# user_input = [sym.strip() for sym in user_input]


def map_user_input(user_input):
    mapped_input = []
    for symptom in user_input:
        if symptom in phrase_to_symptom:
            mapped_input.append(phrase_to_symptom[symptom])
        else:
            # Try fuzzy match
            close_matches = difflib.get_close_matches(symptom, phrase_to_symptom.keys(), n=1, cutoff=0.8)
            if close_matches:
                matched = close_matches[0]
                print(f"🤖 Interpreted '{symptom}' as '{matched}'")
                mapped_input.append(phrase_to_symptom[matched])
            else:
                print(f"⚠️ Warning: '{symptom}' not recognized. Will use it as-is.")
                mapped_input.append(symptom)
    return mapped_input


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

        # Create input vector
        input_vector = [1 if symptom in user_input else 0 for symptom in all_symptoms]

        # Predict
        prediction = model.predict([input_vector])[0]

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



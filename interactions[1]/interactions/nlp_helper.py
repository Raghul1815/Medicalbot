import spacy
import joblib
import json
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm")

# Load the trained model if it exists
try:
    model = joblib.load('chatbot/ml_model.pkl')
    vectorizer = joblib.load('chatbot/vectorizer.pkl')
except FileNotFoundError:
    model = None
    vectorizer = None

# Analyze user input to extract intent using basic NLP techniques
def analyze_input(user_input):
    doc = nlp(user_input.lower())
    
    # Define a simple rule-based intent recognition based on lemmas
    for token in doc:
        if token.lemma_ in ['hello', 'hi', 'hey']:
            return "greeting"
        elif token.lemma_ == 'bye':
            return "farewell"
        elif token.lemma_ in ['good', 'well', 'fine', 'okay', 'great']:
            return "well_being"
        elif token.lemma_ in ['bad', 'sad', 'unwell', 'tired']:
            return "not_well"
        elif token.lemma_ in ['help', 'support']:
            return "request_help"
        elif token.lemma_ in ['thank', 'thanks']:
            return "gratitude"
        elif token.lemma_ in ['how', 'are', 'you']:
            return "check_wellbeing"
        elif token.lemma_ in ['what', 'is', 'your', 'name']:
            return "ask_name"
        elif token.lemma_ in ['tell', 'me', 'about']:
            return "request_info"
        elif token.lemma_ in ['fever', 'instructions']:
            return "fever"
        elif token.lemma_ in ['chemical', 'instructions']:
            return "chemical_burns"
        elif token.lemma_ in ['drowning', 'instructions']:
            return "drowning"
        elif token.lemma_ in ['electric', 'instructions']:
            return "electric_shock"
        elif token.lemma_ in ['allergic', 'instructions']:
            return "allergic_reactions"
        elif token.lemma_ in ['eye', 'instructions']:
            return "eye_injury"
        elif token.lemma_ in ['hypoglycemia', 'instructions']:
            return "hypoglycemia"
        elif token.lemma_ in ['hyperglycemia', 'instructions']:
            return "hyperglycemia"
        elif token.lemma_ in ['drug', 'instructions']:
            return "drug_overdose"
        elif token.lemma_ in ['nose', 'instructions']:
            return "nose_injury"
        elif token.lemma_ in ['throat', 'instructions']:
            return "throat_injury"
        elif token.lemma_ in ['rib', 'instructions']:
            return "rib_injury"
        elif token.lemma_ in ['pelvic', 'instructions']:
            return "pelvic_injury"
        elif token.lemma_ in ['hip', 'instructions']:
            return "hip_dislocation"
        elif token.lemma_ in ['shoulder', 'instructions']:
            return "shoulder_dislocation"
        elif token.lemma_ in ['knee', 'instructions']:
            return "knee_dislocation"
        elif token.lemma_ in ['finger', 'instructions']:
            return "finger_dislocation"
        elif token.lemma_ in ['torn', 'instructions']:
            return "torn_ligament"
        elif token.lemma_ in ['alcohol', 'instructions']:
            return "alcohol_poisoning"
        elif token.lemma_ in ['chest', 'imstructions']:
            return "chest_pain"
        elif token.lemma_ in ['dental', 'instructions']:
            return "dental_emergencies"
        elif token.lemma_ in ['ear', 'instructions']:
            return "ear_injury"
        elif token.lemma_ in ['insect', 'instructions']:
            return "insect_stings"
        elif token.lemma_ in ['animal', 'instructions']:
            return "animal_bites"
        elif token.lemma_ in ['snake', 'instructions']:
            return "snake_bite"
        elif token.lemma_ in ['spider', 'instructions']:
            return "spider_bite"
        elif token.lemma_ in ['scorpion', 'instructions']:
            return "scorpion_sting"
        elif token.lemma_ in ['jellyfish', 'instructions']:
            return "jellyfish_sting"
        elif token.lemma_ in ['tick', 'instructions']:
            return "tick_bite"
        elif token.lemma_ in ['human', 'instructions']:
            return "human_bite"
        elif token.lemma_ in ['foreign', 'instructions']:
            return "foreign_object_in_the_eye"
        elif token.lemma_ in ['heat', 'instructions']:
            return "heat_exhaustion"
        elif token.lemma_ in ['cold', 'instructions']:
            return "cold_exposure"
        elif token.lemma_ in ['panic', 'instructions']:
            return "panic_attack"
        elif token.lemma_ in ['shock', 'instructions']:
            return "shock"
        elif token.lemma_ in ['sepsis', 'instructions']:
            return "sepsis"
        elif token.lemma_ in ['sunburn', 'instructions']:
            return "sunburn"
        elif token.lemma_ in ['blisters', 'instructions']:
            return "blisters"
        elif token.lemma_ in ['abrasions', 'instructions']:
            return "abrasions"
        elif token.lemma_ in ['hyperventilation', 'instructions']:
            return "hyperventilation"
        elif token.lemma_ in ['gallstones', 'instructions']:
            return "gallstones"
        elif token.lemma_ in ['sickle', 'instructions']:
            return "sickle_cell_crisis"
        elif token.lemma_ in ['blood', 'instructions']:
            return "blood_clot"
        elif token.lemma_ in ['hypotension', 'instructions']:
            return "hypotension"
        elif token.lemma_ in ['earwax', 'instructions']:
            return "earwax_blockage"
        elif token.lemma_ in ['herniated', 'instructions']:
            return "herniated_disc"
        elif token.lemma_ in ['pneumonia', 'instructions']:
            return "pneumonia"
        elif token.lemma_ in ['anaphylaxis', 'instructions']:
            return "anaphylaxis"
        elif token.lemma_ in ['poisoning', 'instructions']:
            return "poisoning"
        elif token.lemma_ in ['deep', 'instructions']:
            return "deep_vein_thrombosis"
        elif token.lemma_ in ['heart', 'instructions']:
            return "heart_palpitations"
        elif token.lemma_ in ['hypertension', 'instructions']:
            return "hypertension"
        elif token.lemma_ in ['ear', 'instructions']:
            return "ear_infection"
        elif token.lemma_ in ['eye', 'instructions']:
            return "eye_infection"
        elif token.lemma_ in ['appendicitis', 'instructions']:
            return "appendicitis"
        elif token.lemma_ in ['gallbladder', 'instructions']:
            return "gallbladder"
        elif token.lemma_ in ['kidney', 'instructions']:
            return "kidney_stones"
        elif token.lemma_ in ['urinary', 'instructions']:
            return "urinary_tract_infection"
        elif token.lemma_ in ['dehydration', 'instructions']:
            return "dehydration"
        elif token.lemma_ in ['food', 'instructions']:
            return "food-poisoning"
        elif token.lemma_ in ['muscle', 'instructions']:
            return "muscle_cramps"
        elif token.lemma_ in ['tendonitis', 'instructions']:
            return "tendonitis"
        elif token.lemma_ in ['gallbladder', 'instructions']:
            return "gallbladder_pain"
        elif token.lemma_ in ['sciatica', 'instructions']:
            return "sciatica"
        elif token.lemma_ in ['testicular', 'instructions']:
            return "testicular_torsion"
        elif token.lemma_ in ['ovarian', 'instructions']:
            return "ovarian_cyst"
        elif token.lemma_ in ['ectopic', 'instructions']:
            return "ectopic_pregnancy"
        elif token.lemma_ in ['miscarriage', 'instructions']:
            return "miscarriaage"
        elif token.lemma_ in ['labor', 'instructions']:
            return "labor_pain"
        elif token.lemma_ in ['preterm', 'instructions']:
            return "preterm_labor"
        elif token.lemma_ in ['menstrual', 'instructions']:
            return "menstrual_cramps"
        elif token.lemma_ in ['dental', 'instructions']:
            return "dental_abscess"
        elif token.lemma_ in ['tooth', 'instructions']:
            return "tooth_fracture"
        elif token.lemma_ in ['corneal', 'instructions']:
            return "corneal_abrasion"
        elif token.lemma_ in ['nail', 'instructions']:
            return "nail_bed_injury"
        elif token.lemma_ in ['ring', 'instructions']:
            return "ring_avulsion"
        elif token.lemma_ in ['gangrene', 'instructions']:
            return "gangrene"
        elif token.lemma_ in ['ulcers', 'instructions']:
            return "ulcers"
        elif token.lemma_ in ['hypertensive', 'instructions']:
            return "hypertensive_emergency"
        elif token.lemma_ in ['pulmonary', 'instructions']:
            return "pulmonary_embolism"
        elif token.lemma_ in ['anxiety', 'instructions']:
            return "anxiety_attack"
        elif token.lemma_ in ['high', 'instructions']:
            return "high_fever"
        elif token.lemma_ in ['choking', 'instructions']:
            return "choking"
        elif token.lemma_ in ['bleeding', 'instructions']:
            return "bleeding"
        elif token.lemma_ in ['hypothermia', 'instructions']:
            return "hypothermia"
        elif token.lemma_ in ['seizure', 'instructions']:
            return "seizure"
        elif token.lemma_ in ['stroke', 'instructions']:
            return "stroke"
        elif token.lemma_ in ['heat', 'instructions']:
            return "heat_stroke"
        elif token.lemma_ in ['asthma', 'instructions']:
            return "asthma_attack"
        elif token.lemma_ in ['heart', 'instructions']:
            return "heart_attack"
        elif token.lemma_ in ['diabetic', 'instructions']:
            return "diabetic_0emergency"
        elif token.lemma_ in ['bleedind', 'instructions']:
            return "bleeding(severe)"
        elif token.lemma_ in ['nosebleeds', 'instructions']:
            return "nosebleeds"
        elif token.lemma_ in ['broken', 'instructions']:
            return "broken_bones"
        elif token.lemma_ in ['sprains', 'instructions']:
            return "sprains_and_strains"
        elif token.lemma_ in ['concussion', 'instructions']:
            return "concussion"
        elif token.lemma_ in ['head', 'instructions']:
            return "head_injury"
        elif token.lemma_ in ['nose', 'instructions']:
            return "nose_fracture"
        elif token.lemma_ in ['body', 'instructions']:
            return "body_in_airway"
        elif token.lemma_ in ['swallowed', 'instructions']:
            return "swallowed_object"
        elif token.lemma_ in ['thyroid', 'instructions']:
            return "thyroid_emergencies"
        elif token.lemma_ in ['insect', 'instructions']:
            return "insect_stings"


# Predict intent using a machine learning model
def predict_intent(user_input):
    global model, vectorizer
    
    if model is None or vectorizer is None:
        print("Model not found. Please train the model.")
        return analyze_input(user_input)  # Fallback to rule-based analysis

    # Transform input using the vectorizer
    input_vector = vectorizer.transform([user_input])
    
    # Make prediction
    predicted = model.predict(input_vector)
    
    return predicted[0]  # Return the predicted intent

# Function to train the model
def train_model():
    global model, vectorizer
    
    # Load data from interactions.json (update this to your actual data source)
    with open('chatbot/json/interactions.json', 'r') as file:
        interactions = json.load(file)

    inputs = [interaction["input"] for interaction in interactions["data"]]
    outputs = [interaction["intent"] for interaction in interactions["data"]]

    # Vectorize the inputs
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(inputs)
    y = outputs

    # Train the model
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X, y)

    # Save the model and vectorizer
    joblib.dump(model, 'chatbot/ml_model.pkl')
    joblib.dump(vectorizer, 'chatbot/vectorizer.pkl')

if __name__ == "__main__":
    # Example usage
    user_input = "Hi there!"
    print("Predicted Intent:", predict_intent(user_input))

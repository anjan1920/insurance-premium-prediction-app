import streamlit as st
import pickle
import os
import pandas as pd

# Get absolute path to models folder
MODEL_FOLDER = os.path.join(os.path.dirname(__file__), "models")

# Load Model trained on age <= 25
with open(os.path.join(MODEL_FOLDER, "poly_model_pipeline.pkl"), "rb") as f:
    model_young = pickle.load(f)

# Load Model trained on age > 25
with open(os.path.join(MODEL_FOLDER, "reg_model_rest.pkl"), "rb") as f:
    model_rest = pickle.load(f)

# Load scaler
with open(os.path.join(MODEL_FOLDER, "minmax_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

st.title("🏥 Health Insurance Prediction App")

# ---------- MAPPING DICTIONARIES ----------
medical_score_map = {
    'No Disease': 0,
    'Thyroid': 1,
    'High blood pressure': 2,
    'Diabetes': 3,
    'Heart disease': 4,
    'Diabetes & Thyroid': 4,
    'Diabetes & High blood pressure': 5,
    'High blood pressure & Heart disease': 6,
    'Diabetes & Heart disease': 7
}

insurance_plan_map = {
    'Bronze': 1,
    'Silver': 2,
    'Gold': 3
}

employment_status_map = {
    'Freelancer': 1,
    'Self-Employed': 2,
    'Salaried': 3
}

smoking_status_map = {
    'Regular': 3,      
    'Occasional': 2,  
    'No Smoking': 1            
}

# ---------- UI INPUTS ----------
with st.container():
    st.markdown("### Enter Your Details")
    st.markdown("---")

    age = st.number_input("Age", min_value=18, max_value=99)
    number_of_dependants = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    region = st.selectbox("Region", ["Northwest", "Southeast", "Northeast", "Southwest"])
    marital_status = st.selectbox("Marital Status", ["Unmarried", "Married"])
    bmi_category = st.selectbox("BMI Category", ["Normal", "Obesity", "Overweight", "Underweight"])
    smoking_status = st.selectbox("Smoking Status", ["No Smoking", "Regular", "Occasional"])
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Freelancer"])
    medical_history = st.selectbox("Medical History", list(medical_score_map.keys()))
    insurance_plan = st.selectbox("Insurance Plan", ["Bronze", "Silver", "Gold"])
    income = st.number_input("Annual Income (in Lakh ₹)", min_value=1.00, max_value=100.0, step=0.1)

# ---------- BUTTON ----------
if st.button("Predict"):
    # Apply mappings
    medical_score = medical_score_map.get(medical_history, 0)
    insurance_plan_numerical = insurance_plan_map.get(insurance_plan, 0)
    employment_score = employment_status_map.get(employment_status, 0)
    smoking_score = smoking_status_map.get(smoking_status, 0)
    gender_male = 1 if gender == "Male" else 0
    marital_status_unmarried = 1 if marital_status == "Unmarried" else 0

    # Region one-hot encoding
    region_Northwest = 1 if region == "Northwest" else 0
    region_Southeast = 1 if region == "Southeast" else 0
    region_Southwest = 1 if region == "Southwest" else 0
    # Northeast baseline

    # BMI category one-hot
    bmi_Obesity = 1 if bmi_category == "Obesity" else 0
    bmi_Overweight = 1 if bmi_category == "Overweight" else 0
    bmi_Underweight = 1 if bmi_category == "Underweight" else 0
    # Normal baseline

    # Prepare model input in the exact order
    model_input_dict = {
        "age": age,
        "number_of_dependants": number_of_dependants,
        "income_lakhs": income,
        "medical_score": medical_score,
        "insurance_plan_numerical": insurance_plan_numerical,
        "employment_score": employment_score,
        "smoking_score": smoking_score,
        "region_Northwest": region_Northwest,
        "region_Southeast": region_Southeast,
        "region_Southwest": region_Southwest,
        "marital_status_Unmarried": marital_status_unmarried,
        "gender_Male": gender_male,
        "bmi_category_Obesity": bmi_Obesity,
        "bmi_category_Overweight": bmi_Overweight,
        "bmi_category_Underweight": bmi_Underweight
    }

    # Print to terminal
    print("\n--- Model Input in Order ---")
    for key, value in model_input_dict.items():
        print(f"{key}: {value}")
    
        # Columns to scale
    # Model input order
    model_columns = [
        'age', 'number_of_dependants', 'income_lakhs',
        'medical_score', 'insurance_plan_numerical',
        'employment_score', 'smoking_score',
        'region_Northwest', 'region_Southeast', 'region_Southwest',
        'marital_status_Unmarried', 'gender_Male',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight'
    ]
    cols_to_scale = [
        'age', 'number_of_dependants', 'income_lakhs',
        'medical_score', 'employment_score',
        'smoking_score', 'insurance_plan_numerical'
    ]

    # Create DataFrame with one row, ensuring correct column order
    input_df = pd.DataFrame([[model_input_dict[col] for col in model_columns]], columns=model_columns)

    # Scale only the numeric columns
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    # Convert to numpy array for model
    scaled_input = input_df.values

    # Prediction
    predicted_amount = 0

    if model_input_dict["age"] <= 25:
        # Directly predict — pipeline applies PolynomialFeatures + LinearRegression
        predicted_amount = model_young.predict(scaled_input)
    else:
        predicted_amount = model_rest.predict(scaled_input)

    print("Predicted amount:", predicted_amount[0])



    # Show in UI too
    predicted_value = round(predicted_amount[0], 1)
    st.success(f"Predicted Amount ₹{predicted_value}")
    # Show model performance summary based on age group
    if model_input_dict["age"] > 25:
        st.info("**Model Performance (age > 25)**\n"
                "✅ Accuracy ~95%\n"
                "📊 ~78% predictions within ±10% error")
    else:
        st.warning("**Model Performance (age ≤ 25)**\n"
                "⚠ Accuracy ~60%\n"
                "📊 Limited predictions due to smaller dataset")
    

    #now load the model 

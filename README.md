# Disease Prediction System  

An interactive Python-based machine learning project that predicts diseases from symptoms using a **Decision Tree Classifier** trained on medical datasets. The system provides disease descriptions, severity analysis, and precautionary measures. It also saves user data and diagnosis history for future reference.  

---

## Features  

- **Symptom-based Prediction**: Predicts diseases using a trained Decision Tree model.  
- **Interactive Diagnosis**: Asks users about symptoms and duration to refine predictions.  
- **Condition Severity Check**: Calculates whether symptoms require doctor consultation.  
- **Detailed Information**: Provides descriptions and precautionary measures for predicted diseases.  
- **User Management**: Saves user profile (name, age, history) in `user_data.json`.  
- **Data-driven**: Uses datasets for symptoms, severity, descriptions, and precautions.  

---

## Project Structure  

```bash
├── Data/
│   ├── Training.csv
│   └── Testing.csv
├── MasterData/
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── symptom_severity.csv
├── user_data.json
├── disease_prediction.py   # main script
bash`

## Requirements
Python 3.8+
pandas
numpy
scikit-learn


## Install dependencies:
```bash
pip install pandas numpy scikit-learn

Usage

## Clone the repository
```bash
git clone https://github.com/your-username/disease-prediction-system.git
cd disease-prediction-system


## Run the program
```bash
python disease_prediction.py


## Follow the prompts

Enter your name and age (first-time users).
Input your symptoms and duration.
Get a predicted disease, description, and precautions.
```bash
Example Interaction
-----------------------------------Disease Prediction-----------------------------------
Your Name: John
Your Age: 25
Hello, John
=====================================================================

Enter the symptom you are experiencing: fever
searches related to input:
0 ) continuous_fever
Select the one you meant (0 - 0):  0
Okay. From how many days ?: 3
...
You may have Malaria
Description: Malaria is caused by...
Take following measures:
1) Consult a doctor
2) Take prescribed medication
3) Rest and stay hydrated
4) Use mosquito repellent

## Dataset

**Training.csv / Testing.csv: Contain symptom-disease mappings.
**symptom_Description.csv: Disease descriptions.
**symptom_severity.csv: Severity values for symptoms.
**symptom_precaution.csv: Precautionary steps for each disease.


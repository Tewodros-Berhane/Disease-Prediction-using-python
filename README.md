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

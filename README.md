# Heart Disease Prediction using Machine Learning

## Project Overview

This project predicts whether a patient has heart disease based on clinical features such as age, sex, blood pressure, cholesterol, chest pain type, and more. It demonstrates classification using a RandomForest model with hyperparameter optimization via GridSearchCV.

## Dataset

- **Source:** [Cleveland Heart Disease Dataset – Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Features Include:**
  - Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope, number of major vessels, thal
- **Target:** 1 = heart disease, 0 = no heart disease

## Tech Stack

- **Python** - Core programming language
- **Pandas, NumPy** - Data manipulation and numerical computations
- **Scikit-learn** - Machine learning library (RandomForestClassifier, GridSearchCV)
- **Matplotlib, Seaborn** - Data visualization

## Project Structure

```
heart-disease-prediction/
│
├── README.md                        # Project documentation
├── heart_disease_prediction.ipynb   # Main notebook with analysis
├── heart.csv                        # Dataset
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Ignore checkpoints and unwanted files
└── checkpoints/                     # (optional) For storing trained models
```

## Notebook Workflow

1. **Import Libraries** – Load all necessary Python libraries
2. **Load Dataset** – Read CSV and explore initial data
3. **Exploratory Data Analysis (EDA)** – Visualize features, correlations, and target distribution
4. **Preprocessing** – Feature scaling, train/test split
5. **Model Training** – RandomForestClassifier with GridSearchCV for hyperparameter tuning
6. **Evaluation** – Accuracy, classification report, confusion matrix
7. **Conclusion & Insights** – Key results and future improvements

## Results

- **Accuracy:** ~85–90% on the test set (depending on split)
- **Confusion matrix** visualizes true positives, true negatives, false positives, and false negatives
- **GridSearchCV** identified optimal hyperparameters for the RandomForest model

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hexa01/heart-disease-prediction.git
   ```

2. **Navigate to project folder:**
   ```bash
   cd heart-disease-prediction
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook:**
   ```bash
   jupyter notebook heart_disease_prediction.ipynb
   ```

## Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

## Future Improvements

- Test additional ML models (Logistic Regression, XGBoost)
- Implement feature selection and dimensionality reduction
- Deploy as a web app for real-world use
- Add model interpretability using SHAP or LIME
- Cross-validation for more robust performance evaluation
- Feature engineering to improve model performance

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Cleveland Heart Disease Dataset from Kaggle
- Scikit-learn documentation and community
- Open source machine learning community

---

**Note:** This project is for educational and research purposes. Always consult healthcare professionals for medical decisions.

# Car-price-prediction
This project aims to predict the price of used cars using a large dataset from Kaggle and GPU-accelerated machine learning models.

📊 Dataset
Source: 1.2m Used Car Listings

Format: CSV with features like Year, Mileage, Make, Model, City, State, and Price.

🧠 Models Used
cuML Logistic Regression

cuML Random Forest Classifier

XGBoost Regressor (GPU)

Feature scaling using cuML.StandardScaler

🔧 Installation
Before running the notebook, install the necessary packages:

bash
Copy
Edit
pip install kagglehub cudf cuml xgboost scikit-learn joblib
⚙️ How It Works
Data Loading

Downloads the dataset using kagglehub

Loads the CSV into a GPU-compatible cuDF dataframe

Preprocessing

Handles null values

Encodes categorical variables

Scales numeric features (Year, Mileage)

Visualization

Histograms of car years

Mileage vs. price scatterplot

Boxplots of price distribution by make

Correlation heatmap

Model Training

Trains multiple ML models with GPU acceleration

Evaluates using metrics like RMSE and R²

Saving Models

Trained models are saved with joblib for later use

📈 Example Visualization
📍 Price vs Mileage

📍 Price by Top 10 Makes

📍 Correlation Matrix of Key Features

🧪 Evaluation Metrics
RMSE (Root Mean Squared Error)

R² Score

🧠 Prediction
Accepts user input for car details

Outputs predicted price using the trained model

📁 File Structure
text
Copy
Edit
Musab_Project_ML.ipynb        # Main notebook
/scaler.joblib                # Scaler for feature normalization
/xgb_model.joblib             # Trained XGBoost model
/random_forest_model.joblib   # Trained Random Forest model

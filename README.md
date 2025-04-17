# Car Price Prediction

This project aims to predict the price of used cars using a large dataset from Kaggle and GPU-accelerated machine learning models.

## 📊 Dataset

- **Source**: 1.2M Used Car Listings
- **Format**: CSV with features like Year, Mileage, Make, Model, City, State, and Price.

## 🧠 Models Used

- cuML Logistic Regression
- cuML Random Forest Classifier
- XGBoost Regressor (GPU)
- Feature scaling using cuML.StandardScaler

## 🔧 Installation

Before running the notebook, install the necessary packages:

```bash
pip install kagglehub cudf cuml xgboost scikit-learn joblib
```

## ⚙️ How It Works

### Data Loading
- Downloads the dataset using kagglehub
- Loads the CSV into a GPU-compatible cuDF dataframe

### Preprocessing
- Handles null values
- Encodes categorical variables
- Scales numeric features (Year, Mileage)

### Visualization
- Histograms of car years
- Mileage vs. price scatterplot
- Boxplots of price distribution by make
- Correlation heatmap

### Model Training
- Trains multiple ML models with GPU acceleration
- Evaluates using metrics like RMSE and R²

### Saving Models
- Trained models are saved with joblib for later use

## 📈 Example Visualizations

### Correlation Matrix
![Correlation Matrix](https://github.com/username/car-price-prediction/raw/main/images/correlation_matrix.png)

This correlation matrix shows the relationships between key variables:
- Strong negative correlation (-0.818) between Year and Mileage (newer cars have lower mileage)
- Moderate positive correlation between Year and Price (newer cars tend to cost more)
- Negative correlation (-0.331) between Mileage and Price (higher mileage typically means lower price)

### Scatter and Density Plot
![Scatter and Density Plot](https://github.com/username/car-price-prediction/raw/main/images/scatter_density_plot.png)

This comprehensive visualization shows relationships between all variables with both scatter plots and distribution curves:
- The diagonal shows distribution of each variable
- Off-diagonal elements show relationships between pairs of variables
- Correlation coefficients are displayed for each relationship

### Feature Distributions
![Feature Distributions](https://github.com/username/car-price-prediction/raw/main/images/feature_distributions.png)

Key observations:
- Year distribution shows most cars in the dataset are from 2012-2016
- State distribution shows highest representation from FL, TX, and CA
- Model distribution shows "LX" as the most common model designation

## 🧪 Evaluation Metrics

- RMSE (Root Mean Squared Error)
- R² Score

## 🧠 Prediction

- Accepts user input for car details
- Outputs predicted price using the trained model

## 📁 File Structure

```
Musab_Project_ML.ipynb        # Main notebook
/scaler.joblib                # Scaler for feature normalization
/xgb_model.joblib             # Trained XGBoost model
/random_forest_model.joblib   # Trained Random Forest model
/images/                      # Visualization images
```

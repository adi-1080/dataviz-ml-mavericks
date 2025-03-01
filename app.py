import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Load the data
@st.cache_data
def load_data():
    climate_df = pd.read_csv("climate_change_data.csv")
    deforestation_df = pd.read_csv("goal15.forest_shares.csv")
    return climate_df, deforestation_df

climate_df, deforestation_df = load_data()

# Function to convert ISO3 codes to country names
def iso3_to_country(iso_code):
    try:
        return pycountry.countries.get(alpha_3=iso_code).name
    except (AttributeError, LookupError):
        return None

# Preprocess the deforestation data
deforestation_df['Country'] = deforestation_df['iso3c'].apply(iso3_to_country)
deforestation_df_updated = deforestation_df.dropna(subset=['Country', 'trend'])

# Merge the data
merged_df = climate_df.merge(deforestation_df_updated, on='Country', how='inner')

# Preprocess the climate data
merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.date

# Streamlit app
st.title("Climate Change and Deforestation Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

# Filter by Country
country_list = merged_df['Country'].unique()
# Add "All Countries" option to the list
country_list = np.insert(country_list, 0, "All Countries")
selected_country = st.sidebar.selectbox("Select Country", country_list)

# Filter by Year
year_list = merged_df['Date'].apply(lambda x: x.year).unique()
selected_year = st.sidebar.selectbox("Select Year", sorted(year_list))

# Filter by Temperature Range
min_temp = merged_df['Temperature'].min()
max_temp = merged_df['Temperature'].max()
temp_range = st.sidebar.slider("Select Temperature Range", min_temp, max_temp, (min_temp, max_temp))

# Filter by Deforestation Trend
min_trend = merged_df['trend'].min()
max_trend = merged_df['trend'].max()
trend_range = st.sidebar.slider("Select Deforestation Trend Range", min_trend, max_trend, (min_trend, max_trend))

# Apply filters (updated to handle "All Countries")
filtered_df = merged_df[
    ((merged_df['Country'] == selected_country) | (selected_country == "All Countries")) &
    (merged_df['Date'].apply(lambda x: x.year) == selected_year) &
    (merged_df['Temperature'] >= temp_range[0]) & (merged_df['Temperature'] <= temp_range[1]) &
    (merged_df['trend'] >= trend_range[0]) & (merged_df['trend'] <= trend_range[1])
]

# Display filtered data
st.write("### Filtered Data")
if filtered_df.empty:
    st.warning("No data found for the selected filters.")
else:
    st.write(filtered_df)

# Visualizations
st.write("### Visualizations")

# Temperature vs Deforestation Trend
if not filtered_df.empty:
    st.write("#### Temperature vs Deforestation Trend")
    fig, ax = plt.subplots()
    sns.scatterplot(x=filtered_df['trend'], y=filtered_df['Temperature'], ax=ax)
    ax.set_xlabel("Deforestation Trend (Negative = Loss)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Relationship Between Deforestation and Temperature Increase")
    st.pyplot(fig)

# Correlation Heatmap
if not filtered_df.empty:
    st.write("#### Correlation Heatmap")
    corr = filtered_df[['Temperature', 'trend', 'CO2 Emissions', 'Sea Level Rise', 'Precipitation', 'Humidity', 'Wind Speed']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# Top 10 Countries with Highest Deforestation
st.write("#### Top 10 Countries with Highest Deforestation")
ranked_deforestation = deforestation_df_updated.sort_values(by="trend").head(10)
fig, ax = plt.subplots()
sns.barplot(x=ranked_deforestation['trend'], y=ranked_deforestation['Country'], ax=ax)
ax.set_xlabel("Deforestation Trend")
ax.set_ylabel("Country")
ax.set_title("Top 10 Countries with Highest Deforestation")
st.pyplot(fig)

# Top 10 Countries with Lowest Deforestation
st.write("#### Top 10 Countries with Lowest Deforestation")
ranked_deforestation = deforestation_df_updated.sort_values(by="trend", ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(x=ranked_deforestation['trend'], y=ranked_deforestation['Country'], ax=ax)
ax.set_xlabel("Deforestation Trend")
ax.set_ylabel("Country")
ax.set_title("Top 10 Countries with Lowest Deforestation")
st.pyplot(fig)

# Machine Learning Models
st.write("### Machine Learning Models")

# Select Features for Prediction
features = ['Temperature', 'forests_2000', 'forests_2020', 'Sea Level Rise', 'Precipitation', 'Humidity', 'Wind Speed']
target = 'CO2 Emissions'

# Train-Test Split
X = merged_df[features]
y = merged_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define Models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': lgb.LGBMRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'ADA Boost': AdaBoostRegressor(),
    'K Neighbors Regressor': KNeighborsRegressor(),
    'Linear SVR': LinearSVR(),
}

# Train & Evaluate Models
results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Store predictions for comparison
    predictions[name] = y_pred
    
    # Model Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MAE': mae, 'MSE': mse, 'R² Score': r2}
    
# Convert results to DataFrame
results_df = pd.DataFrame(results).T
st.write("#### Model Evaluation Results")
st.write(results_df)

# Plot Actual vs. Predicted CO2 Emissions
st.write("#### Actual vs. Predicted CO2 Emissions")
fig, ax = plt.subplots(figsize=(14, 8))
for name, y_pred in predictions.items():
    ax.scatter(y_test, y_pred, alpha=0.5, label=name)
ax.plot(y_test, y_test, color='red', linestyle='--', label="Actual (Reference Line)")
ax.set_xlabel("Actual CO2 Emissions")
ax.set_ylabel("Predicted CO2 Emissions")
ax.set_title("Actual vs. Predicted CO2 Emissions for All Models")
ax.legend()
st.pyplot(fig)

# Predict Future CO2 Emissions (Next 10 Years)
st.write("#### Predicted CO2 Emissions for Next 10 Years")
future_years = np.array(range(2025, 2035)).reshape(-1, 1)
future_X = X_test.sample(len(future_years), replace=True)  # Reuse test samples

future_predictions = {}
for name, model in models.items():
    future_predictions[name] = model.predict(future_X)

# Plot Future CO2 Emissions Trend
fig, ax = plt.subplots(figsize=(14, 8))
for name, y_pred in future_predictions.items():
    ax.plot(future_years, y_pred, marker='o', linestyle='-', label=name)
ax.set_xlabel("Year")
ax.set_ylabel("Predicted CO2 Emissions")
ax.set_title("Predicted CO2 Emissions for Next 10 Years (All Models)")
ax.legend()
st.pyplot(fig)

# Display raw data
if st.checkbox("Show Raw Data"):
    st.write("### Raw Data")
    st.write(merged_df)
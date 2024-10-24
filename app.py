# Create directory if it doesn't exist
import os
os.makedirs('.streamlit', exist_ok=True)

# Write the configuration
with open('.streamlit/config.toml', 'w') as f:
    f.write('[theme]\n')
    f.write('primaryColor = "#575fe8"\n')
    f.write('backgroundColor = "#f5f7f3"\n')
    f.write('secondaryBackgroundColor = "#c9eab8"\n')
    f.write('textColor = "#262730"\n')
    f.write('font = "sans serif"\n')
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load the trained Decision Tree model
with open('decision_tree_model.pkl', 'rb') as f:
    dtpipeline = pickle.load(f)

# Load the dataset for FLW visualizations
flw_data = pd.read_csv("FLWData.csv")

# Remove rows with NaN values in 'loss_percentage' and 'food_supply_stage'
flw_data = flw_data.dropna(subset=['loss_percentage', 'food_supply_stage'])

# Set Streamlit app title and description
st.title("WasteWise")
st.write(
    "This app predicts the **percentage of food loss and waste (FLW)** based on commodity, country, year, and supply chain stage. This app uses a **decision tree** model trained on data from the United Nations Food Loss and Waste Database: https://www.fao.org/platform-food-loss-waste/flw-data/en/ "
)

# --- Prediction Section ---

st.header("Decision Tree FLW Predictions ")

# User input for country
country = st.selectbox("Select Country", flw_data['country'].unique(), key='country_select')

# Filter the dataset based on selected country
filtered_data_by_country = flw_data[flw_data['country'] == country]

# Dynamically filter the commodities and supply chain stages based on the selected country
commodities_for_country = filtered_data_by_country['commodity'].unique()
supply_chain_stages_for_country = filtered_data_by_country['food_supply_stage'].unique()

# User inputs for commodity and supply chain stage, which are filtered based on country
commodity = st.selectbox("Select Commodity", commodities_for_country, key='commodity_select')
supply_chain_stage = st.selectbox("Select Supply Chain Stage", supply_chain_stages_for_country, key='supply_chain_stage_select')

# User input for year
year = st.number_input(
    "Select Input Year",
    min_value=int(flw_data['year'].min()),
    max_value=int(flw_data['year'].max()),
    step=1,
    key='year_input'
)

# Predict button
if st.button("Predict FLW Percentage"):
    input_data = pd.DataFrame({
        'country': [country],
        'commodity': [commodity],
        'year': [year],
        'food_supply_stage': [supply_chain_stage]
    })

    prediction = dtpipeline.predict(input_data)[0]
    st.write(f"Predicted Food Loss and Waste Percentage: {prediction:.2f}%")

# Function to wrap text
def wrap_text(text, max_length):
    if len(text) > max_length:
        words = text.split()
        wrapped_lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_length:
                current_line += word + " "
            else:
                wrapped_lines.append(current_line.strip())
                current_line = word + " "

        wrapped_lines.append(current_line.strip())
        return "\n".join(wrapped_lines)
    return text

# Visualization options
st.write("## Decision Tree FLW Visualizations")

viz_option = st.selectbox(
    "Choose a visualization:",
    ("Predicted FLW by Commodity", "Predicted FLW by Supply Chain Stage"),
    key='viz_option_select'
)

# Button to generate the FLW by Commodity graph
if viz_option == "Predicted FLW by Commodity":
    if st.button("Generate FLW by Commodity Graph"):
        predictions_by_commodity = []
        for com in commodities_for_country:
            input_data = pd.DataFrame({
                'country': [country],
                'commodity': [com],
                'year': [year],
                'food_supply_stage': [supply_chain_stage]
            })
            prediction = dtpipeline.predict(input_data)[0]
            predictions_by_commodity.append((com, prediction))

        # Sort and display the top 10 commodities
        top_10_commodities = sorted(predictions_by_commodity, key=lambda x: x[1], reverse=True)[:10]
        top_10_com_names, top_10_pred_values = zip(*top_10_commodities)

        wrapped_com_names = [wrap_text(name, 20) for name in top_10_com_names]

        # Plot the top 10 commodities
        plt.figure(figsize=(10, 5))
        plt.bar(wrapped_com_names, top_10_pred_values, color='lightgreen')

        plt.title(f"Top 10 Predicted FLW by Commodity for {country} in {year} at {supply_chain_stage} Stage")
        plt.xlabel("Commodity")
        plt.ylabel("Predicted FLW Percentage")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

# Button to generate the FLW by Supply Chain Stage graph
if viz_option == "Predicted FLW by Supply Chain Stage":
    if st.button("Generate FLW by Supply Chain Stage Graph"):
        predictions_by_stage = []
        for stage in supply_chain_stages_for_country:
            input_data = pd.DataFrame({
                'country': [country],
                'commodity': [commodity],
                'year': [year],
                'food_supply_stage': [stage]
            })
            prediction = dtpipeline.predict(input_data)[0]
            predictions_by_stage.append((stage, prediction))

        sorted_stages = sorted(predictions_by_stage, key=lambda x: x[1], reverse=True)
        stage_names, pred_values = zip(*sorted_stages)

        wrapped_stage_names = [wrap_text(name, 20) for name in stage_names]

        # Plot the predicted FLW by supply chain stage
        plt.figure(figsize=(10, 5))
        plt.bar(wrapped_stage_names, pred_values, color='lightgreen')

        plt.title(f"Predicted FLW by Supply Chain Stage for {commodity} in {country} in {year}")
        plt.xlabel("Supply Chain Stage")
        plt.ylabel("Predicted FLW Percentage")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

# --- Real Data Visualization Section ---

st.header("FLW Observed vs. Predicted Values")
st.write(
    "This section visualizes the **actual percentage of food loss and waste (FLW)** according to real data from the dataset and plots it alongside the predicted FLW values from the decision tree."
)

# User input for country
country_real = st.selectbox("Select Country", flw_data['country'].unique(), key='country_real_select')

# Filter the dataset based on selected country
filtered_data_by_country_real = flw_data[flw_data['country'] == country_real]

# Dynamically filter the commodities based on the selected country
commodities_for_country_real = filtered_data_by_country_real['commodity'].unique()

# User input for commodity
commodity_real = st.selectbox("Select Commodity", commodities_for_country_real, key='commodity_real_select')

# Filter the dataset based on selected commodity
filtered_data_by_commodity_real = filtered_data_by_country_real[filtered_data_by_country_real['commodity'] == commodity_real]

# Dynamically filter the years and supply chain stages based on the selected commodity
years_for_commodity_real = filtered_data_by_commodity_real['year'].unique()
supply_chain_stages_for_commodity_real = filtered_data_by_commodity_real['food_supply_stage'].unique()

# User inputs for year and supply chain stage
year_real = st.selectbox("Select Year", years_for_commodity_real, key='year_real_select')
supply_chain_stage_real = st.selectbox("Select Supply Chain Stage", supply_chain_stages_for_commodity_real, key='supply_chain_stage_real_select')

# Visualization options
st.write("## Decision Tree Predictions Vs. Observed FLW Values")

viz_option_real = st.selectbox(
    "Choose a visualization for comparison:",
    ("FLW by Commodity", "FLW by Supply Chain Stage"),
    key='viz_option_real_select'
)

# Function to predict FLW using the decision tree model
def predict_flw(country, commodity, year, supply_chain_stage):
    input_data = pd.DataFrame({
        'country': [country],
        'commodity': [commodity],
        'year': [year],
        'food_supply_stage': [supply_chain_stage]
    })
    return dtpipeline.predict(input_data)[0]

# Button to generate the FLW by Commodity graph for actual data
if viz_option_real == "FLW by Commodity":
    if st.button("Generate FLW by Commodity Comparison"):
        filtered_data_by_year_stage_real = filtered_data_by_commodity_real[
            (filtered_data_by_commodity_real['year'] == year_real) &
            (filtered_data_by_commodity_real['food_supply_stage'] == supply_chain_stage_real)
        ]

        # Group by commodity and calculate average loss percentage
        avg_loss_by_commodity_real = filtered_data_by_year_stage_real.groupby('commodity')['loss_percentage'].mean().sort_values(ascending=False)

        # Get predicted values
        predicted_values_real = [predict_flw(country_real, com, year_real, supply_chain_stage_real) for com in avg_loss_by_commodity_real.index]

        # Plot the loss percentage by commodity alongside predicted values
        plt.figure(figsize=(10, 5))
        bar_width = 0.35
        index = range(len(avg_loss_by_commodity_real))

        bar1 = plt.bar(index, avg_loss_by_commodity_real.values, bar_width, label='Actual FLW Percentage', color='orange')
        bar2 = plt.bar([i + bar_width for i in index], predicted_values_real, bar_width, label='Predicted FLW Percentage', color='lightgreen')

        plt.title(f"Actual vs Predicted FLW by Commodity for {country_real} in {year_real} at {supply_chain_stage_real} Stage")
        plt.xlabel("Commodity")
        plt.ylabel("FLW Percentage")
        plt.xticks([i + bar_width / 2 for i in index], avg_loss_by_commodity_real.index, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

# Button to generate the FLW by Supply Chain Stage graph for actual data
if viz_option_real == "FLW by Supply Chain Stage":
    if st.button("Generate FLW by Supply Chain Stage Comparison"):
        filtered_data_by_commodity_year_real = filtered_data_by_country_real[
            (filtered_data_by_country_real['commodity'] == commodity_real) &
            (filtered_data_by_country_real['year'] == year_real)
        ]

        # Group by supply chain stage and calculate average loss percentage
        avg_loss_by_stage_real = filtered_data_by_commodity_year_real.groupby('food_supply_stage')['loss_percentage'].mean().sort_values(ascending=False)

        # Get predicted values
        predicted_values_stage_real = [predict_flw(country_real, commodity_real, year_real, stage) for stage in avg_loss_by_stage_real.index]

        # Plot the loss percentage by supply chain stage alongside predicted values
        plt.figure(figsize=(10, 5))
        bar_width = 0.35
        index = range(len(avg_loss_by_stage_real))

        bar1 = plt.bar(index, avg_loss_by_stage_real.values, bar_width, label='Actual FLW Percentage', color='orange')
        bar2 = plt.bar([i + bar_width for i in index], predicted_values_stage_real, bar_width, label='Predicted FLW Percentage', color='lightgreen')

        plt.title(f"Actual vs Predicted FLW by Supply Chain Stage for {commodity_real} in {country_real} in {year_real}")
        plt.xlabel("Supply Chain Stage")
        plt.ylabel("FLW Percentage")
        plt.xticks([i + bar_width / 2 for i in index], avg_loss_by_stage_real.index, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set the application title
st.title('We ♥️ Machine Learning')

# Display information about the application
st.info('Deploy Python ML models with Streamlit')

# Importing and displaying raw data
with st.expander('Data'):
    st.write('**Raw Data**')
    # Load the dataset
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
    st.write(df)  # Display the dataset

    # Display feature matrix (X)
    st.write('**X**')
    X_raw = df.drop('species', axis=1)  # Remove the target column to form features
    st.write(X_raw)

    # Display target vector (Y)
    st.write('**Y**')
    Y_raw = df.species  # Define the target variable
    st.write(Y_raw)

# Visualization of data
with st.expander('Visualizations'):
    # Create a scatter plot to visualize the relationship between bill length and body mass
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Collecting input features from the user
with st.sidebar:
    st.header('Input features')
    # Input selectors for features
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    gender = st.selectbox('Gender', ('male', 'female'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body Mass (g)', 2700.0, 6400.0, 4207.0)

    # Prepare input features dataframe
    data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df = pd.DataFrame(data, index=[0])
    # Concatenate user input with raw data for consistent encoding
    input_penguins = pd.concat([input_df, X_raw], axis=0)

# Display user input and combined data
with st.expander('Input features'):
    st.write('**Input penguin**')
    st.write(input_df)  # Display user input
    st.write('**Combined Penguin Data**')
    st.write(input_penguins)  # Display combined data

# Data preparation
# Encode categorical features
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
input_row = df_penguins[:1]  # Extract encoded user input row
X = df_penguins[1:]  # Prepare feature matrix

# Encode target labels
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]

Y = Y_raw.apply(target_encode)

# Display encoded data
with st.expander('Data preparation'):
    st.write('**Encoded X input penguin**')
    st.write(input_row)  # Display encoded features for user input
    st.write('**Encoded Y**')
    st.write(Y)  # Display encoded target

# Model training and inference
# Initialize and train the RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Make predictions on user input
prediction = clf.predict(input_row)
prediction_prob = clf.predict_proba(input_row)

# Display prediction probabilities
df_prediction_prob = pd.DataFrame(prediction_prob)
df_prediction_prob.columns = ['Adelie', 'Chinstrap', 'Gentoo']  # Add species labels as columns

# Display probabilities as a styled dataframe
st.dataframe(df_prediction_prob,
             column_config={
                 'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', width='medium', min_value=0, max_value=1),
                 'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', width='medium', min_value=0, max_value=1),
                 'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', width='medium', min_value=0, max_value=1)
             }, hide_index=True)

# Display the predicted species
st.subheader('Predicted Species')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))

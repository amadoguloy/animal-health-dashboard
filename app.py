import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import glob

# Load multiple dataset chunks
file_paths = glob.glob("/mnt/data/animal_health_chunk_*.csv")
df = pd.concat([pd.read_csv(fp) for fp in file_paths])

# Preprocessing: Convert categorical variables into numerical labels
label_encoders = {}
categorical_columns = ['World region', 'Country', 'Administrative Division', 'Disease', 'Animal Category']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Prepare features and target variable
df['Outbreak_Flag'] = df['New outbreaks'].apply(lambda x: 1 if str(x).isdigit() and int(x) > 0 else 0)
features = ['Year', 'World region', 'Country', 'Disease', 'Animal Category']
X = df[features]
y = df['Outbreak_Flag']

# Train a predictive model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
model_path = "/mnt/data/animal_health_model.pkl"
joblib.dump(model, model_path)

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Animal Health Disease Outbreak Dashboard"),
    
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': c, 'value': c} for c in df['Country'].unique()],
        placeholder="Select a Country"
    ),
    
    dcc.Graph(id='outbreak-trend'),
    
    html.H2("Predict Outbreak Risk"),
    dcc.Dropdown(id='predict-region', options=[{'label': c, 'value': c} for c in df['World region'].unique()], placeholder="Select Region"),
    dcc.Dropdown(id='predict-country', options=[{'label': c, 'value': c} for c in df['Country'].unique()], placeholder="Select Country"),
    dcc.Dropdown(id='predict-disease', options=[{'label': c, 'value': c} for c in df['Disease'].unique()], placeholder="Select Disease"),
    dcc.Dropdown(id='predict-animal', options=[{'label': c, 'value': c} for c in df['Animal Category'].unique()], placeholder="Select Animal Category"),
    dcc.Input(id='predict-year', type='number', placeholder='Enter Year', value=2025),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output')
])

@app.callback(
    Output('outbreak-trend', 'figure'),
    Input('country-dropdown', 'value')
)
def update_trend(selected_country):
    if selected_country:
        filtered_df = df[df['Country'] == selected_country]
    else:
        filtered_df = df
    fig = px.line(filtered_df, x='Year', y='New outbreaks', title='Outbreak Trends Over Time')
    return fig

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [Input('predict-year', 'value'), Input('predict-region', 'value'), Input('predict-country', 'value'),
     Input('predict-disease', 'value'), Input('predict-animal', 'value')]
)
def predict_outbreak(n_clicks, year, region, country, disease, animal):
    if n_clicks > 0:
        input_data = pd.DataFrame([[year, region, country, disease, animal]], columns=features)
        input_data = input_data.apply(lambda col: label_encoders[col.name].transform(col) if col.name in label_encoders else col)
        prediction = model.predict(input_data)[0]
        return f"Predicted Outbreak: {'Yes' if prediction == 1 else 'No'}"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)

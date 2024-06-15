# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from flask import Flask

# Import your custom modules
import cleaning_module
from cleaning_module import clean_data
import visualization_Module_2
from visualization_Module_2 import main
import regression
from regression import main_, plot_model_comparisons_plotly
import joblib
import pandas as pd

# Load the Linear Regression model


# +
# Load and clean data
df = clean_data()

# visualization data
topic_counts, fig_cramers_heatmap, CKD1_0_fig, COPD1_1_fig, COPD1_2_fig, CVD1_4_fig, CVD1_2_fig, CVD1_3_fig, CVD1_5_fig, CVD2_0_fig, AST3_1_fig, CVD3_1_fig, CVD3_data_fig, COPD5_1_fig, COPD5_4_fig, \
race_mor_diab_fig, race_mor_copd_fig, race_mor_can_fig, race_mor_ast_fig, gen_mor_diab_fig, gen_mor_ast_fig, gen_mor_can_fig, gen_mor_copd_fig, gen_mor_cvd_fig, \
figsoda_comsumption, cvd_fig, immunization_heatmap_fig, old_adult_data_fig, fig_oral_health_map = visualization_Module_2.main(df)


# +
# Create a Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

model = joblib.load('linear_regression_model.pkl')
# Load and clean data
# df = clean_data()

# # visualization data
# topic_counts, fig_cramers_heatmap, CKD1_0_fig, COPD1_1_fig, COPD1_2_fig, CVD1_4_fig, CVD1_2_fig, CVD1_3_fig, CVD1_5_fig, CVD2_0_fig, AST3_1_fig, CVD3_1_fig, CVD3_data_fig, COPD5_1_fig, COPD5_4_fig, \
# race_mor_diab_fig, race_mor_copd_fig, race_mor_can_fig, race_mor_ast_fig, gen_mor_diab_fig, gen_mor_ast_fig, gen_mor_can_fig, gen_mor_copd_fig, gen_mor_cvd_fig, \
# figsoda_comsumption, cvd_fig, immunization_heatmap_fig, old_adult_data_fig, fig_oral_health_map = visualization_Module_2.main(df)

# regression data
lr_mse, lr_r2, lr_r2_train,xgb_mse, xgb_r2, xgb_r2_train,rf_mse, rf_r2,rf_r2_train,lr_predictions, xgb_predictions, rf_predictions,X_test, y_test = regression.main_(df)
comparison_graphs = regression.plot_model_comparisons_plotly(X_test, y_test,lr_predictions, xgb_predictions, rf_predictions)


# Dropdown options for Mortality, Hospitalization, Stratification, and Additional Insights
mortality_options = [
    {'label': 'Mortality with end-stage renal disease', 'value': 'CKD1_0'},
    {'label': 'Mortality as underlying cause among adults aged >= 45 years', 'value': 'COPD1_1'},
    {'label': 'Mortality as underlying or contributing cause among adults aged >= 45 years', 'value': 'COPD1_2'},
    {'label': 'Mortality from heart failure', 'value': 'CVD1_4'},
    {'label': 'Mortality from coronary heart disease', 'value': 'CVD1_2'},
    {'label': 'Mortality from diseases of the heart', 'value': 'CVD1_3'},
    {'label': 'Mortality from cerebrovascular disease', 'value': 'CVD1_5'},
]

hospitalization_options = [
    {'label': 'Hospitalizations for Asthma', 'value': 'AST3_1'},
    {'label': 'Hospitalizations for stroke', 'value': 'CVD3_1'},
    {'label': 'Hospitalization for heart failure among Medicare-eligible Patients', 'value': 'CVD2_0'},
    {'label': 'Hospitalization for acute myocardial infarction', 'value': 'CVD3_data'},
    {'label': 'Hospitalization for COPD as first-listed diagnosis', 'value': 'COPD5_1'},
    {'label': 'Hospitalization for COPD as any diagnosis among Medicare-eligible persons', 'value': 'COPD5_4'}
]

stratification_options = [
    {'label': 'Race Distribution of Mortality Number in Diabetes', 'value': 'race_mor_diab'},
    {'label': 'Race Distribution of Mortality Number in COPD', 'value': 'race_mor_copd'},
    {'label': 'Race Distribution of Mortality Number in Cancer', 'value': 'race_mor_can'},
    {'label': 'Race Distribution of Hospitalization Number in Cardiovascular Disease', 'value': 'race_mor_ast'},
    {'label': 'Race Distribution of Hospitalization Number in Diabetes', 'value': 'gen_mor_diab'},
    {'label': 'Gender Distribution of Mortality Number in Asthma', 'value': 'gen_mor_ast'},
    {'label': 'Gender Distribution of Mortality Number in Cancer', 'value': 'gen_mor_can'},
    {'label': 'Gender Distribution of Mortality Number in COPD', 'value': 'gen_mor_copd'},
    {'label': 'Gender Distribution of Mortality Number in CVD', 'value': 'gen_mor_cvd'},
]

additional_insights_options = [
    {'label': 'Yearly trend of Soda consumption among high school students', 'value': 'soda'},
    {'label': 'Top 10 States with Highest Cardiovascular Disease Count', 'value': 'cvd'},
    {'label': 'Immunization distribution', 'value': 'imm'},
    {'label': 'Gender wise older adults aged up to date on a core set of clinical preventive services', 'value': 'old'},
    {'label': 'Oral Health, Cancer, Reproductive Health, Diabetes Distribution', 'value': 'can'},
]

# Insight labels

Mortality_insights = {
    'CVD1_4': 'California, Florida, and Texas have the highest index for mortality from heart failure. This could either be due to one of few reasons, the population density, or geography',
    'CVD1_2': 'California, Florida, Texas and New York all seem to have the highest index for mortality from coronary heart disease. This could be a result of the population density in those 4 states',
    'CVD1_3': 'The same 4 states (California, Texas, Florida and New York) have the highest index for mortality from total cardiovascular disease as well',
    'CVD1_5': 'Here, California, Texas and Florida have drastically higher indexes for mortality from cerebrovascular disease (stroke)',
    'CKD1_0': 'Philadelphia seems to have a higher rate of mortality after California and Texas',
    'COPD1_1': 'There seems to be a lot of outlier data in Kansas state (Crude Rate nearing to 300%), A lot of other states also have high crude rates.',
    'COPD1_2': 'Ohio has a high mortality rate in this division after (California, Texas, Florida)'
}

Hospitalization_insights = {
    'AST3_1': 'Hospitalizations for Asthma have decreased over the years, Although the decline is not linear, We can safely say, that total count has decreased over the years',
    'CVD3_1': 'We see a linear decline in the hospitalization rate till the year 2013 and we also see a drop in the year 2015, but after that the hospitalization rate has increased.',
    'CVD2_0': 'Due to the expansion of medicare eligibility umbrella in 2014, we see a steep incline in hospitalization rates among this group since the year 2014.',
    'CVD3_data': 'We can see a steep decline in crude rate of hospitalization in the year 2014 (19 to 15%)',
    'COPD5_1': 'There is a decline in the hospitalization rates till the year 2015 and then we see a slight incline in the hospitalization numbers in this category',
    'COPD5_4': 'We see an incline in hospitalization numbers after 2014'
}

Stratification_insights = {
    'race_mor_diab': 'White, non-Hispanic people have a significantly higher index and Black, non-Hispanic people and Hispanic people seem to be around the same amount.',
    'race_mor_copd': 'Although White, non-Hispanic people seem to have the higher index for most charts, it is safe to say it is an extreme outlier here with it representing over 80% of the chart.',
    'race_mor_can': 'Here we can see that although typically Black, non-Hispanic people and Hispanic people seem to have around the same percentage, here, Black, non-Hispanic people are almost 2x larger than Hispanic people.',
    'race_mor_ast': 'Mortality number in Asthma here is almost evenly distributed among the 4 present races, with American Indian or Alaska Native not being represented in the chart at all.',
    'gen_mor_diab': 'Here, Males seem to be a little higher than Female for Diabetes which can be true as men typically seem to be diagnosed with diabetes more often than women.',
    'gen_mor_ast': 'Although typically, the charts seem to be quite evenly distributed for both genders, here we can see that Females seem to be almost 2/3 of the chart.',
    'gen_mor_can': 'Mortality number in cancer is quite evenly distributed for both genders. It can be assumed that both genders were evenly studied and/or are affected just about the same amount.',
    'gen_mor_copd': 'Although, most graphs were evenly distributed for the genders, this seems to be the closest to a 50/50 split.',
    'gen_mor_cvd': 'Here, we can see that the mortality for CVD is quite evenly distributed amongst both genders.',
}

Additional_insights = {
    'soda': 'There is a decline in soda consumption among high school students over the years. This indicates that the students are option for healtier options than soda',
    'cvd': 'California, florida, texas and newyork have the highest count of overall cardiovascular diseases.',
    'imm': 'South Dakota and oklahoma states have the highest immunization related disease count among all the states in USA.',
    'old': 'Male population is general are more upto date on clinical preventive services than the female population',
    'can':'Interestingly, South dakota has higher crude prevelance thatn california for oral cancer from 2008 to 2010 '
}


# Dropdown components for Mortality, Hospitalization, Stratification, and Additional Insights
mortality_dropdown = dcc.Dropdown(
    options=mortality_options,
    value=mortality_options[0]['value'],
    placeholder='Select a value',
    id='mortality-selector'
)

hospitalization_dropdown = dcc.Dropdown(
    options=hospitalization_options,
    value=hospitalization_options[0]['value'],
    placeholder='Select a value',
    id='hospitalization-selector'
)

stratification_dropdown = dcc.Dropdown(
    options=stratification_options,
    value=stratification_options[0]['value'],
    placeholder='Select a value',
    id='stratification-selector'
)

additional_insights_dropdown = dcc.Dropdown(
    options=additional_insights_options,
    value=additional_insights_options[0]['value'],
    placeholder='Select a value',
    id='additional-insights-selector'
)

# Define the layout of the application with pages and tabs
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Initial Analysis', children=[
            html.H1(children='Chronic Disease Analysis in USA', style={'textAlign': 'center', 'font-family': 'Times New Roman', 'font-weight': 'bold'}),
            html.Marquee("Health is wealth", style={'font-family': 'Times New Roman', 'color': 'blue', 'font-size': '20px'}),
            html.H3('Initial Analysis of Data', style={'font-family': 'Times New Roman'}),
            html.P('The below given graphs give us the distribution of data in Chronic disease dataset', style={'font-family': 'Times New Roman'}),
            dcc.Graph(figure=topic_counts),
            html.P('Cancer seems to be the highest reported Chronic disease followed by Cardiovascular disease, COPD and Diabeties', style={'font-family': 'Times New Roman'}),
            html.P('The Below given heat map explains the correlation between different disease topics', style={'font-family': 'Times New Roman'}),
            dcc.Graph(figure=fig_cramers_heatmap),
            html.P('The heatmap shows that there are strong correlations between many of the chronic diseases. CVD is having the strongest correlation of 0.99 among all diseases. '
                   'For Instanse, cardiovascular disease is strongly correlated with diabetes (0.56), chronic kidney disease (0.39), and obesity (0.47). ' 
                   'Diabetes is also strongly correlated with chronic kidney disease (0.21) and obesity (0.39).', style={'font-family': 'Times New Roman'})
        ]),
        
        dcc.Tab(label='Data Visualization', children=[
            html.H1(children='Chronic Disease Analysis in USA', style={'textAlign': 'center', 'font-family': 'Times New Roman', 'font-weight': 'bold'}),
            
            dcc.Tabs([
                dcc.Tab(label='Mortality Trends', children=[
                    dbc.Row([
                        dbc.Col([
                            html.H5('Select Mortality Trend'),
                            dcc.Dropdown(id='mortality-selector', options=mortality_options, value=mortality_options[0]['value'])
                        ], md=4),
                        dbc.Col([
                            dcc.Graph(id='mortality-graph'),
                            html.Div(id='mortality-insights-container')
                        ], md=8)
                    ])
                ]),
                
                dcc.Tab(label='Hospitalization Trends', children=[
                    dbc.Row([
                        dbc.Col([
                            html.H5('Select Hospitalization Trend'),
                            dcc.Dropdown(id='hospitalization-selector', options=hospitalization_options, value=hospitalization_options[0]['value'])
                        ], md=4),
                        dbc.Col([
                            dcc.Graph(id='hospitalization-graph'),
                            html.Div(id='hospitalization-insights-container')
                        ], md=8)
                    ])
                ]),
                
                dcc.Tab(label='Stratification Trends', children=[
                    dbc.Row([
                        dbc.Col([
                            html.H5('Select Stratification Trend'),
                            dcc.Dropdown(id='stratification-selector', options=stratification_options, value=stratification_options[0]['value'])
                        ], md=4),
                        dbc.Col([
                            dcc.Graph(id='stratification-graph'),
                            html.Div(id='stratification-insights-container')
                        ], md=8)
                    ])
                ]),
                
                dcc.Tab(label='Additional Insights', children=[
                    dbc.Row([
                        dbc.Col([
                            html.H5('Select Additional Insights'),
                            dcc.Dropdown(id='additional-insights-selector', options=additional_insights_options, value=additional_insights_options[0]['value'])
                        ], md=4),
                        dbc.Col([
                            dcc.Graph(id='additional-insights-graph'),
                            html.Div(id='additional-insights-container')
                        ], md=8)
                    ])
                ]),
            ])
        ]),
        
        dcc.Tab(label='Regression Model', children=[
            html.H1(children='Chronic Disease Analysis in USA', style={'textAlign': 'center', 'font-family': 'Times New Roman', 'font-weight': 'bold'}),
            html.H3(children='Prediction of Alcohol use among youth in the coming years', style={'font-family': 'Times New Roman'}),
            html.P('We employed three distinct regression models in our analysis: Linear Regression, XGBOOST, and Random Forest. '
                   'Among these, the Linear Regression model demonstrated superior performance, evidenced by the highest R-squared value and '
                   'the smallest disparity between training and test R-squared values. The accompanying diagram illustrates the model fit for all three regression models. '
                   "It's evident that the Linear Regression model aligns most closely with the majority of data points, thereby establishing it as the most effective model for our predictions.", style={'font-family': 'Times New Roman'}),
            html.Div([
                dcc.Graph(figure=comparison_graphs),
                dcc.Input(id='yearstart', type='number', placeholder='Prediction Year')
            ]),
            html.Button('Predict', id='predict-button', n_clicks=0),
            html.Div(id='prediction-output', style={'margin-top': '20px'})
        ]),
    ]),
])

@app.callback(
    Output('mortality-graph', 'figure'),
    [Input('mortality-selector', 'value')]
)
def update_mortality_graph(selected_value):
    if selected_value == 'CKD1_0':
        return CKD1_0_fig
    elif selected_value == 'COPD1_1':
        return COPD1_1_fig
    elif selected_value == 'COPD1_2':
        return COPD1_2_fig
    elif selected_value == 'CVD1_4':
        return CVD1_4_fig
    elif selected_value == 'CVD1_2':
        return CVD1_2_fig
    elif selected_value == 'CVD1_3':
        return CVD1_3_fig
    elif selected_value == 'CVD1_5':
        return CVD1_5_fig

@app.callback(
    Output('hospitalization-graph', 'figure'),
    [Input('hospitalization-selector', 'value')]
)
def update_hospitalization_graph(selected_value):
    if selected_value == 'AST3_1':
        return AST3_1_fig
    elif selected_value == 'CVD3_1':
        return CVD3_1_fig
    elif selected_value == 'CVD2_0':
        return CVD2_0_fig
    elif selected_value == 'CVD3_data':
        return CVD3_data_fig
    elif selected_value == 'COPD5_1':
        return COPD5_1_fig
    elif selected_value == 'COPD5_4':
        return COPD5_4_fig

@app.callback(
    Output('stratification-graph', 'figure'),
    [Input('stratification-selector', 'value')]
)
def update_stratification_graph(selected_value):
    if selected_value == 'race_mor_diab':
        return race_mor_diab_fig
    elif selected_value == 'race_mor_copd':
        return race_mor_copd_fig
    elif selected_value == 'race_mor_can':
        return race_mor_can_fig
    elif selected_value == 'race_mor_ast':
        return race_mor_ast_fig
    elif selected_value == 'gen_mor_diab':
        return gen_mor_diab_fig
    elif selected_value == 'gen_mor_ast':
        return gen_mor_ast_fig
    elif selected_value == 'gen_mor_can':
        return gen_mor_can_fig
    elif selected_value == 'gen_mor_copd':
        return gen_mor_copd_fig
    elif selected_value == 'gen_mor_cvd':
        return gen_mor_cvd_fig

@app.callback(
    Output('additional-insights-graph', 'figure'),
    [Input('additional-insights-selector', 'value')]
)
def update_additional_insights_graph(selected_value):
    if selected_value == 'soda':
        return figsoda_comsumption
    elif selected_value == 'cvd':
        return cvd_fig
    elif selected_value == 'imm':
        return immunization_heatmap_fig
    elif selected_value == 'old':
        return old_adult_data_fig
    elif selected_value == 'can':
        return fig_oral_health_map

@app.callback(
    Output('mortality-insights-container', 'children'),
    [Input('mortality-selector', 'value')]
)
def update_mortality_insights(selected_value):
    return dcc.Markdown(Mortality_insights.get(selected_value, 'Select a mortality trend to see insights.'))

# Callback to update hospitalization insights
@app.callback(
    Output('hospitalization-insights-container', 'children'),
    [Input('hospitalization-selector', 'value')]
)
def update_hospitalization_insights(selected_value):
    return dcc.Markdown(Hospitalization_insights.get(selected_value, 'Select a hospitalization trend to see insights.'))

# Callback to update stratification insights
@app.callback(
    Output('stratification-insights-container', 'children'),
    [Input('stratification-selector', 'value')]
)
def update_stratification_insights(selected_value):
    return dcc.Markdown(Stratification_insights.get(selected_value, 'Select a stratification trend to see insights.'))

# Callback to update additional insights
@app.callback(
    Output('additional-insights-container', 'children'),
    [Input('additional-insights-selector', 'value')]
)
def update_additional_insights(selected_value):
    return dcc.Markdown(Additional_insights.get(selected_value, 'Select an additional insight to see insights.'))

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('yearstart', 'value')]
)

def predict(n_clicks, yearstart):  # Adjust parameters if needed
    if n_clicks > 0 and yearstart is not None:
        # Assuming yearend is the same as yearstart
        # Add default values for other fields if they are not input by the user
        input_df = pd.DataFrame({
            'yearstart': [yearstart],
            'yearend': [yearstart],  # Assuming yearend is the same as yearstart
            'locationabbr': ['US'],  # Replace with actual default
            'stratification1': ['Overall']  # Replace with actual default
        })

        # Make prediction
        prediction = model.predict(input_df)[0]
        return f'Predicted crude prevalence in the input year: {prediction:.2f}'
    return ''
    
    
    
# Run the application
if __name__ == '__main__':
    app.run_server(debug=True, port=8081)
# -



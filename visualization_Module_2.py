import pandas as pd
# Example usage with a DataFrame named 'your_dataframe'
# Replace 'your_dataframe' with the actual name of your DataFrame
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
import plotly.graph_objects as go

def main(df):
        
    # Function for treemap
    def treemap(categories, title, path, values):
    
        fig = px.treemap(categories, path=path, values=values, height=700,
                         title=title, color_discrete_sequence=px.colors.sequential.RdBu)
        fig.data[0].textinfo = 'label+text+value'
        return fig
    
    # Function for bar_graph
    def bar_graph(data, x, y, color, title, xaxis, yaxis):
        fig = px.bar(data, x=x, y=y, color=color, labels={x: xaxis, y: yaxis})
        fig.update_layout(
            title_text=title,
            xaxis_title_text=xaxis,
            yaxis_title_text=yaxis,
            bargap=0.2,
            bargroupgap=0.1
        )
        fig.update_xaxes(categoryorder='total descending')
        return fig
    
    # Function for line_chart
    def line_chart(data, x, y, color, title, xaxis, yaxis):
        fig = px.line(data, x=x, y=y, color=color, labels={x: xaxis, y: yaxis},
                      title=title)
        return fig
    
    # Function for box_plot
    def box_plot(data, x, y, color, title, xaxis, yaxis):
        fig = px.box(data, x=x, y=y, color=color, labels={x: xaxis, y: yaxis})
        fig.update_layout(
            title_text=title,
            xaxis_title_text=xaxis,
            yaxis_title_text=yaxis,
        )
        return fig
        
    # Function for pie_chart
    def pie_chart(data, names, values, title):
        fig = px.pie(data, names=names, values=values, title=title)
        return fig
        
    # Function for heatmap
    def heatmap(data, locations, color, color_continuous_scale, title, labels):
        fig = px.choropleth(data,
                            locations=locations,
                            locationmode="USA-states",
                            color=color,
                            color_continuous_scale=color_continuous_scale,
                            title=title,
                            labels=labels,
                            scope="usa")
        return fig
        
    
    # ---------------------------------------------------------------------------INITIAL ANALYSIS---------------------------------------------------------------------------    
    
    # Count of Each Topic in the Dataset
    def topic_count(df):
        topic_counts = df['topic'].value_counts().sort_values(ascending=False).reset_index()
        topic_counts.columns = ['Topic', 'Count']
    
    # Using the custom function to create a bar graph
        topic_counts = bar_graph(topic_counts, 'Count', 'Topic', 'Count', 'Count of Each Topic in the Dataset', 'Count', 'Topics')
        return topic_counts
    
    # ---------------------------------------------------------------------------Mortality---------------------------------------------------------------------------
    
    # Chronic Kidney Disease - Mortality incidents with end-stage renal disease
    def CKD1_0(df):
        filtered_data = df[(df['topic'] == 'Chronic Kidney Disease')
                            & (df['question'] == 'Mortality with end-stage renal disease')
                            & (df['datavaluetypeid'] == 'NMBR')
                            & (df['locationabbr'] != 'US')]
    
        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'] , errors = 'coerce')
    
        if not filtered_data.empty:
             CKD1_0_fig = box_plot(filtered_data, 'locationabbr', 'datavalue', 'locationabbr',
                     'Statewise distribution of Mortality with end-stage renal disease (Chronic Kidney Disease)',
                     'States', 'Mortality Numbers')
        else:
            print("No data to plot.")
        return CKD1_0_fig
    
    # Chronic Obstructive Pulmonary Disease - Mortality with chronic obstructive pulmonary disease as underlying cause among adults aged >= 45 years
    def COPD1_1(df):
        filtered_data = df[(df['topic'] == 'Chronic Obstructive Pulmonary Disease')
                       & (df['question'] == 'Mortality with chronic obstructive pulmonary disease as underlying cause among adults aged >= 45 years')
                       & (df['datavaluetypeid'] == 'CRDRATE')
                       & (df['locationabbr'] != 'US')]
    
        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'] , errors = 'coerce')
    
        if not filtered_data.empty:
            COPD1_1_fig = box_plot(filtered_data, 'locationabbr', 'datavalue', 'locationabbr',
                     'Statewise distribution of Mortality with chronic obstructive pulmonary disease as underlying cause among adults aged >= 45 years',
                     'States', 'Crude Rate (%)')
        else:
            print("No data to plot.")
        return COPD1_1_fig
    
    # Chronic Obstructive Pulmonary Disease - Mortality with chronic obstructive pulmonary disease as underlying or contributing cause among adults aged >= 45 years
    def COPD1_2(df):
        filtered_data = df[(df['topic'] == 'Chronic Obstructive Pulmonary Disease')
                       & (df['question'] == 'Mortality with chronic obstructive pulmonary disease as underlying or contributing cause among adults aged >= 45 years')
                       & (df['datavaluetypeid'] == 'NMBR')
                       & (df['locationabbr'] != 'US')]
    
        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'] , errors = 'coerce')
    
        if not filtered_data.empty:
            COPD1_2_fig = box_plot(filtered_data, 'locationabbr', 'datavalue', 'locationabbr',
                     'Statewise distribution of Mortality with chronic obstructive pulmonary disease as underlying cause among adults aged >= 45 years',
                     'States', 'Mortality Numbers')
        else:
            print("No data to plot.")
        return COPD1_2_fig
    
    # Cardiovascular Disease - Mortality from heart failure
    def CVD1_4(df):
        filtered_data = df[(df['topicid'] == 'CVD')
                       & (df['questionid'] == 'CVD1_4')
                       & (df['datavaluetypeid'] == 'NMBR')
                       & (df['locationabbr'] != 'US')]
    
        #df.loc[filtered_data.index, 'datavalue'] = pd.to_numeric(filtered_data['datavalue'], errors='coerce')
    
        if not filtered_data.empty:
            CVD1_4_fig = box_plot(filtered_data, 'locationabbr', 'datavalue', 'locationabbr',
                'Statewise distribution of Mortality from heart failure','States', 'Mortality Numbers')
        else:
            print("No data to plot.")
        return CVD1_4_fig
    
    # Cardiovascular Disease - Mortality from coronary heart disease
    def CVD1_2(df):
        filtered_data = df[(df['topicid'] == 'CVD')
                        & (df['questionid'] == 'CVD1_2')
                        & (df['datavaluetypeid'] == 'NMBR')
                        & (df['locationabbr'] != 'US')]
    
        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'] , errors = 'coerce')
    
        if not filtered_data.empty:
            CVD1_2_fig = box_plot(filtered_data, 'locationabbr', 'datavalue', 'locationabbr',
                    'Statewise distribution of Mortality from coronary heart disease','States', 'Mortality Numbers')
        else:
            print("No data to plot.")
        return CVD1_2_fig
    
    # Cardiovascular Disease - Mortality from diseases of the heart
    def CVD1_3(df):
        filtered_data = df[(df['topicid'] == 'CVD')
                        & (df['questionid'] == 'CVD1_3')
                        & (df['datavaluetypeid'] == 'NMBR')
                        & (df['locationabbr'] != 'US')]
    
        #df.loc[filtered_data.index, 'datavalue'] = pd.to_numeric(filtered_data['datavalue'], errors='coerce')
    
        if not filtered_data.empty:
            CVD1_3_fig = box_plot(filtered_data, 'locationabbr', 'datavalue', 'locationabbr',
                    'Statewise distribution of Mortality from diseases of the heart','States', 'Mortality Numbers')
        else:
            print("No data to plot.")
        return CVD1_3_fig 
    # Cardiovascular Disease - Mortality from cerebrovascular disease (stroke)
    def CVD1_5(df):
        filtered_data = df[(df['topicid'] == 'CVD')
                        & (df['questionid'] == 'CVD1_5')
                        & (df['datavaluetypeid'] == 'NMBR')
                        & (df['locationabbr'] != 'US')]
    
        #df.loc[filtered_data.index, 'datavalue'] = pd.to_numeric(filtered_data['datavalue'], errors='coerce')
    
        if not filtered_data.empty:
            CVD1_5_fig = box_plot(filtered_data, 'locationabbr', 'datavalue', 'locationabbr',
                    'Statewise distribution of Mortality from cerebrovascular disease (stroke)','States', 'Mortality Numbers')
        else:
            print("No data to plot.")
    
        return CVD1_5_fig    
    # ---------------------------------------------------------------------------Hospitalization---------------------------------------------------------------------------

        # Hospitalizations for asthma
    def AST3_1(df):
        asthma_data = df[(df['topicid'] == 'AST')
                        & (df['questionid'] == 'AST3_1')
                        & (df['datavaluetype'] == 'Number')
                        & (~df['locationabbr'].isin(['US','PR','GU','VI']))]
    
        yearly_trend = asthma_data.groupby('yearstart')['datavalue'].sum().reset_index()
    
        AST3_1_fig = line_chart(yearly_trend, x='yearstart', y='datavalue', color=None,
                    title='Year-wise Trend of Hospitalizations for Asthma',
                    xaxis='Year', yaxis='Total Hospitalizations')
        return AST3_1_fig
    
    # CVD - Hospitalization for stroke
    def CVD3_1(df):
        CVD1_data = df[(df['topicid'] == 'CVD')
                        & (df['questionid'] == 'CVD3_1')
                        & (df['datavaluetype'] == 'Number')
                        & (~df['locationabbr'].isin(['US','PR','GU','VI']))]
    
        yearly_trend = CVD1_data.groupby('yearstart')['datavalue'].sum().reset_index()
    
        CVD3_1_fig = line_chart(yearly_trend, x='yearstart', y='datavalue', color=None,
                title='Year-wise Trend of Hospitalizations for stroke',
                xaxis='Year', yaxis='Total Hospitalizations')
        return CVD3_1_fig
    
    # cvd - Hospitalization for heart failure among Medicare-eligible persons aged >= 65 years
    def CVD2_0(df):
        CVD2_data = df[(df['topicid'] == 'CVD')
                        & (df['questionid'] == 'CVD2_0')
                        & (df['datavaluetype'] == 'Number')
                        & (~df['locationabbr'].isin(['US','PR','GU','VI']))]
    
        yearly_trend = CVD2_data.groupby('yearstart')['datavalue'].sum().reset_index()
    
        CVD2_0_fig = line_chart(yearly_trend, x='yearstart', y='datavalue', color=None,
                title='Year-wise Trend of Hospitalization for heart failure among Medicare-eligible persons aged >= 65 years',
                xaxis='Year', yaxis='Total Hospitalizations')
        return CVD2_0_fig
    
    # cvd - Hospitalization for acute myocardial infarction
    def CVD3_data(df):
        CVD3_data = df[(df['topicid'] == 'CVD')
                        & (df['questionid'] == 'CVD3_2')
                        & (df['datavaluetypeid'] == 'CRDRATE')
                        & (~df['locationabbr'].isin(['US','PR','GU','VI']))]
    
        yearly_trend = CVD3_data.groupby('yearstart')['datavalue'].mean().reset_index()
    
        CVD3_data_fig = line_chart(yearly_trend, x='yearstart', y='datavalue', color=None,
                   title='Year-wise Trend of Hospitalization for acute myocardial infarction',
                   xaxis='Year', yaxis='Crude Rate (%)')
        return CVD3_data_fig
        
    # copd - Hospitalization for chronic obstructive pulmonary disease as first-listed diagnosis
    def COPD5_1(df):
        copd_data = df[(df['topicid'] == 'COPD')
                        & (df['questionid'] == 'COPD5_1')
                        & (df['datavaluetypeid'] == 'NMBR')
                        & (~df['locationabbr'].isin(['US','PR','GU','VI']))]
    
        yearly_trend = copd_data.groupby('yearstart')['datavalue'].sum().reset_index()
    
        COPD5_1_fig=line_chart(yearly_trend, x='yearstart', y='datavalue', color=None,
                   title='Year-wise Trend of Hospitalization for chronic obstructive pulmonary disease as first-listed diagnosis',
                   xaxis='Year', yaxis='Total Hospitalizations')
        return COPD5_1_fig

    # copd - Hospitalization for chronic obstructive pulmonary disease as any diagnosis among Medicare-eligible persons aged >= 65 years
    def COPD5_4(df):
        copd_data = df[(df['topicid'] == 'COPD')
                        & (df['questionid'] == 'COPD5_4')
                        & (df['datavaluetypeid'] == 'AGEADJRATE')
                        & (~df['locationabbr'].isin(['US','PR','GU','VI']))]
    
        yearly_trend = copd_data.groupby('yearstart')['datavalue'].mean().reset_index()
    
        COPD5_4_fig = line_chart(yearly_trend, x='yearstart', y='datavalue', color=None,
                   title='Year-wise Trend of Hospitalization for chronic obstructive pulmonary disease as any diagnosis among Medicare-eligible persons aged >= 65 years',
                   xaxis='Year', yaxis='age Adjusted rate (per 1000)')
        return COPD5_4_fig

# ---------------------------------------------------------------------------Stratification------------------------------------------------------------------------------------------
    
    #Race Distribution of Mortality Number in Diabetes
    def race_mor_diab(df):
        filtered_data = df[(df['topic'] == 'Diabetes')
                           & (df['question'].isin(['Mortality due to diabetes reported as any listed cause of death',
                                                   'Mortality with diabetic ketoacidosis reported as any listed cause of death'
                                                   ]))
                           & (df['datavaluetypeid'] == 'NMBR')
                           & (df['locationabbr'] != 'US')
                           & (df['stratificationcategoryid1'] == 'RACE')]
    
        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'] , errors = 'coerce')
    
        racewise_mean = filtered_data.groupby('stratification1')['datavalue'].mean().reset_index()
    
        if not racewise_mean.empty:
            race_mor_diab_fig = pie_chart(racewise_mean, 'stratification1', 'datavalue', 'Race Distribution of Mortality Number in Diabetes')
        else:
            print("No data to plot.")
        return race_mor_diab_fig
    
    #Race Distribution of Mortality Number in Chronic Obstructive Pulmonary Disease
    def race_mor_copd(df):
        filtered_data = df[(df['topic'] == 'Chronic Obstructive Pulmonary Disease')
                           & (df['question'].isin(['Mortality with chronic obstructive pulmonary disease as underlying cause among adults aged >= 45 years',
                                                   'Mortality with chronic obstructive pulmonary disease as underlying or contributing cause among adults aged >= 45 years',
                                                   ]))
                           & (df['datavaluetypeid'] == 'NMBR')
                           & (df['locationabbr'] != 'US')
                           & (df['stratificationcategoryid1'] == 'RACE')]
    
        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'] , errors = 'coerce')
    
        racewise_mean = filtered_data.groupby('stratification1')['datavalue'].mean().reset_index()
    
        if not racewise_mean.empty:
            race_mor_copd_fig = pie_chart(racewise_mean, 'stratification1', 'datavalue', 'Race Distribution of Mortality Number in Chronic Obstructive Pulmonary Disease')
        else:
            print("No data to plot.")
        return race_mor_copd_fig

    # Race Distribution of Mortality Number in Cancer
    def race_mor_can(df):
        filtered_data = df[(df['topic'] == 'Cancer')
                           & (df['question'].isin(['Cancer of the oral cavity and pharynx, mortality',
                                                   'Cancer of the prostate, mortality',
                                                   'Invasive cancer (all sites combined), mortality',
                                                   'Invasive cancer of the female breast, incidence' 'Melanoma, mortality',
                                                   'Cancer of the female breast, mortality',
                                                   'Cancer of the female cervix, mortality',
                                                   'Cancer of the colon and rectum (colorectal), mortality',
                                                   'Cancer of the lung and bronchus, mortality']))
                           & (df['datavaluetypeid'] == 'AVGANNNMBR')
                           & (df['locationabbr'] != 'US')
                           & (df['stratificationcategoryid1'] == 'RACE')]
    
        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'] , errors = 'coerce')
    
        racewise_mean = filtered_data.groupby('stratification1')['datavalue'].mean().reset_index()
    
        if not racewise_mean.empty:
            race_mor_can_fig = pie_chart(racewise_mean, 'stratification1', 'datavalue', 'Race Distribution of Mortality Number in Cancer')
        else:
            print("No data to plot.")  
        return  race_mor_can_fig 
    
    # Race Distribution of Mortality Number in Asthma
    def race_mor_ast(df):
        filtered_data = df[(df['topic'] == 'Cardiovascular Disease')
                   & (df['question'].isin(['Hospitalization for stroke',
                                           'Hospitalization for heart failure among Medicare-eligible persons aged >= 65 years',
                                           'Hospitalization for acute myocardial infarction']))
                   & (df['datavaluetypeid'] == 'NMBR')
                   & (df['locationabbr'] != 'US')
                   & (df['stratificationcategoryid1'] == 'RACE')]

        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'], errors='coerce')

        racewise_mean = filtered_data.groupby('stratification1')['datavalue'].mean().reset_index()

        if not racewise_mean.empty:
            # Sort data for better visualization in the funnel chart
            racewise_mean = racewise_mean.sort_values(by='datavalue', ascending=False)

            # Create a funnel chart with custom colorscale
            fig_funnel = go.Figure(go.Funnel(
                y=racewise_mean['stratification1'],
                x=racewise_mean['datavalue'],
                textinfo="value+percent initial",
                marker=dict(colorscale='YlGnBu')
            ))

            # Update layout for the funnel chart
            fig_funnel.update_layout(
                title='Race Distribution of Hospitalization Number in Cardiovascular Disease (Funnel)',
                xaxis_title='Hospitalization Numbers',
                yaxis_title='Race',
                showlegend=False  # Hide legend for simplicity
            )

            # Show the funnel chart
            race_mor_ast_fig = fig_funnel 
            return race_mor_ast_fig
    
    # Gender Distribution of Mortality Number in Diabetes
    def gen_mor_diab(df):
        filtered_data = df[(df['topic'] == 'Diabetes')
                   & (df['question'].isin(['Hospitalization with diabetes as a listed diagnosis']))
                   & (df['datavaluetypeid'] == 'NMBR')
                   & (df['locationabbr'] != 'US')
                   & (df['stratificationcategoryid1'] == 'RACE')]

        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'], errors='coerce')

        racewise_mean = filtered_data.groupby('stratification1')['datavalue'].mean().reset_index()

        if not racewise_mean.empty:
            # Create a sunburst chart
            fig_sunburst = px.sunburst(racewise_mean,
                                       path=['stratification1'],
                                       values='datavalue',
                                       title='Race Distribution of Hospitalization Number in Diabetes (Sunburst)',
                                       color='datavalue',
                                       color_continuous_scale='YlGnBu')

            # Show the sunburst chart
            gen_mor_diab_fig = fig_sunburst
            return gen_mor_diab_fig

        else:
            print("No data to plot.")
        
    # Gender Distribution of Mortality Number in Asthma 
    def gen_mor_ast(df):
        filtered_data = df[(df['topic'] == 'Asthma')
                           & (df['question'].isin(['Asthma mortality rate']))
                           & (df['datavaluetypeid'] == 'NMBR')
                           & (df['locationabbr'] != 'US')
                           & (df['stratificationcategoryid1'] == 'GENDER')]
    
        gender_distribution = filtered_data[['stratification1', 'datavalue']]
    
        if not filtered_data.empty:
            gen_mor_ast_fig = pie_chart(gender_distribution, 'stratification1', 'datavalue', 'Gender Distribution of Mortality Number in Asthma')
        else:
            print("No data to plot.")
        return gen_mor_ast_fig
        
    # Gender Distribution of Mortality Number in Cancer
    def gen_mor_can(df):
        filtered_data = df[(df['topic'] == 'Cancer')
                           & (df['question'].isin(['Cancer of the oral cavity and pharynx, mortality',
                                                   'Cancer of the prostate, mortality',
                                                   'Invasive cancer (all sites combined), mortality',
                                                   'Invasive cancer of the female breast, incidence' 'Melanoma, mortality',
                                                   'Cancer of the female breast, mortality',
                                                   'Cancer of the female cervix, mortality',
                                                   'Cancer of the colon and rectum (colorectal), mortality',
                                                   'Cancer of the lung and bronchus, mortality']))
                           & (df['datavaluetypeid'] == 'AVGANNNMBR')
                           & (df['locationabbr'] != 'US')
                           & (df['stratificationcategoryid1'] == 'GENDER')]
    
        gender_distribution = filtered_data[['stratification1', 'datavalue']]
    
        if not filtered_data.empty:
            gen_mor_can_fig = pie_chart(gender_distribution, 'stratification1', 'datavalue', 'Gender Distribution of Mortality Number in Cancer')
        else:
            print("No data to plot.")
        return gen_mor_can_fig

    # Gender Distribution of Mortality Number in COPD
    def gen_mor_copd(df):
        filtered_data = df[(df['topic'] == 'Chronic Obstructive Pulmonary Disease')
                           & (df['question'].isin(['Mortality with chronic obstructive pulmonary disease as underlying cause among adults aged >= 45 years',
                                                   'Mortality with chronic obstructive pulmonary disease as underlying or contributing cause among adults aged >= 45 years',
                                                   ]))
                           & (df['datavaluetypeid'] == 'NMBR')
                           & (df['locationabbr'] != 'US')
                           & (df['stratificationcategoryid1'] == 'GENDER')]
    
        gender_distribution = filtered_data[['stratification1', 'datavalue']]
    
        if not filtered_data.empty:
            gen_mor_copd_fig = pie_chart(gender_distribution, 'stratification1', 'datavalue', 'Gender Distribution of Mortality Number in Chronic Obstructive Pulmonary Disease')
        else:
            print("No data to plot.")
    
        return gen_mor_copd_fig
    
    # Cardiovascular Disease - Gender Distribution of Mortality numbers
    def gender_dist_cvd_mor(df):
        filtered_data = df[(df['topicid'] == 'CVD')
                        & (df['questionid'].isin(['CVD1_5','CVD1_3','CVD1_2','CVD1_4']))
                        & (df['datavaluetypeid'] == 'NMBR')
                        & (df['locationabbr'] != 'US')
                        & (df['stratificationcategoryid1'] == 'GENDER')]
    
        gender_distribution = filtered_data[['stratification1', 'datavalue']]
    
        if not filtered_data.empty:
            gender_dist_cvd_mor_fig=pie_chart(gender_distribution, 'stratification1', 'datavalue', 'Gender Distribution of Mortality Number')
        else:
            print("No data to plot.")    
        return gender_dist_cvd_mor_fig

    # # -------------------------------------------Additional insights-------------------------------------------

    def soda_comsumption(df):
        filtered_data = df[(df['questionid'] == 'NPAW12_2')
                       & (df['datavaluetypeid'] == 'CRDPREV')
                       & (~df['locationabbr'].isin(['US', 'PR', 'GU', 'VI']))]
    
        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'], errors='coerce')
    
        yearly_trend = filtered_data.groupby('yearstart')['datavalue'].mean().reset_index()
    
        figsoda_comsumption = px.bar(yearly_trend,
                x='yearstart', y='datavalue', color='yearstart',
                labels={'datavalue': 'crude prevelance', 'yearstart': 'Year'})
      
        figsoda_comsumption.update_layout(xaxis = dict(tickmode = 'array', tickvals = [2013, 2015, 2017, 2019]))
        figsoda_comsumption.update_xaxes(categoryorder='total descending')
        figsoda_comsumption = figsoda_comsumption
        return figsoda_comsumption
    
    def cvd_treemap(df):
        cardio_data = df[(df['topic'] == 'Cardiovascular Disease')
                        & (df['datavaluetype'] == 'Number')
                        & (~df['locationabbr'].isin(['US', 'PR', 'GU', 'VI']))]

        data_by_state = cardio_data.groupby('locationdesc')['datavalue'].sum().reset_index()

        cvd_by_state_sorted = data_by_state.sort_values(by='datavalue', ascending=False)

        # Step 4: Select the top 10 states
        top_10_states = cvd_by_state_sorted.head(10)

        cvd_fig = treemap(top_10_states, 'Top 10 States with Highest Cardiovascular Disease Count', ['locationdesc'], 'datavalue')

        return cvd_fig

    def immunization_heatmap(df):
        immunization_df = df[(df['topicid'] == 'IMM')
                            & (df['datavaluetypeid'] == 'AGEADJPREV')
                            & (~df['locationabbr'].isin(['US', 'PR', 'GU', 'VI']))]

        #immunization_df['datavalue'] = pd.to_numeric(immunization_df['datavalue'], errors='coerce')

        # Check if filtered_data is not empty before creating the heatmap
        if not immunization_df.empty:
            immunization_fig = heatmap(immunization_df,
                                    locations='locationabbr',
                                    color='datavalue',
                                    color_continuous_scale='Viridis',
                                    title='Immunization - Age-Adjusted Prevalence by State',
                                    labels={'datavalue': 'Age-Adjusted Prevalence'})
        else:
            print("No data to plot.")

        return immunization_fig

    def old_adult_data_plot(df):
        old_adult_data = df[(df['topicid'] == 'OLD') &
                            (df['questionid'] == 'OLD3_2') &
                            (df['stratificationcategoryid1'] == 'GENDER')]

        male_data = old_adult_data[old_adult_data['stratificationid1'] == 'GENM']
        female_data = old_adult_data[old_adult_data['stratificationid1'] == 'GENF']

        year_avg_male_data = male_data.groupby('yearstart')['datavalue'].mean().reset_index()
        year_avg_female_data = female_data.groupby('yearstart')['datavalue'].mean().reset_index()

        fig_old_adult_data = px.line(title='Gender-wise Proportion of Older Adults Aged 50-64 Years Up to Date on Clinical Preventive Services',
                                    labels={'yearstart': 'Year', 'datavalue': 'Average Crude Prevalence'})
        fig_old_adult_data.add_scatter(x=year_avg_male_data['yearstart'], y=year_avg_male_data['datavalue'], mode='lines', name='Male')
        fig_old_adult_data.add_scatter(x=year_avg_female_data['yearstart'], y=year_avg_female_data['datavalue'], mode='lines', name='Female')
        fig_old_adult_data.update_layout(xaxis_title='Year', yaxis_title='Average Crude Prevalence %')

        return fig_old_adult_data

    def oral_health_map(df):
        oral_df = df[df['topic'].isin(['Oral Health', 'Cancer', 'Reproductive Health', 'Diabetes'])]
        filtered_data = oral_df[(oral_df['yearstart'] >= 2008) & (oral_df['yearstart'] <= 2020)
                                & (oral_df['datavaluetypeid'] == 'CRDPREV')
                                & (~oral_df['locationabbr'].isin(['US','PR', 'GU', 'VI']))]

        #filtered_data['datavalue'] = pd.to_numeric(filtered_data['datavalue'], errors='coerce')

        if not filtered_data.empty:
            fig_oral_health_map = px.choropleth(filtered_data,
                                                locations='locationabbr',
                                                locationmode="USA-states",
                                                color='datavalue',
                                                color_continuous_scale='Plasma',
                                                title='Oral Health, Cancer, Reproductive Health, Diabetes by State from 2008 to 2020',
                                                labels={'datavalue': 'Cancer'},
                                                scope="usa")
            return fig_oral_health_map
        else:
            print("No data to plot.")
            return None
        
    def cramers_heatmap(df):
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

        # Calculate Cramér's V for 'topic' column
        cramers_v_matrix = pd.DataFrame(index=df['topic'].unique(), columns=df['topic'].unique())
        for i in df['topic'].unique():
            for j in df['topic'].unique():
                cramers_v_matrix.loc[i, j] = cramers_v(df['topic'] == i, df['topic'] == j)

        # Convert the matrix values to numeric
        cramers_v_matrix = cramers_v_matrix.apply(pd.to_numeric, errors='coerce')

        # Create a heatmap using Plotly Express
        fig_cramers_heatmap = px.imshow(cramers_v_matrix.values,
                    x=cramers_v_matrix.columns,
                    y=cramers_v_matrix.index,
                    color_continuous_scale="viridis",  # Specify a valid colorscale
                    labels=dict(color="Cramér's V"),
                    title="Correlation Heatmap by Topic - Correlation Matrix")

        # Adjust the size of the heatmap
        fig_cramers_heatmap.update_layout(
            autosize=False,
            width=800,  # Set the desired width
            height=800,  # Set the desired height
            xaxis=dict(tickangle=-45),  # Tilt x-axis labels
        )
        return fig_cramers_heatmap
    
    topic_counts = topic_count(df)
    fig_cramers_heatmap = cramers_heatmap(df)
    CKD1_0_fig = CKD1_0(df)
    COPD1_1_fig = COPD1_1(df)
    COPD1_2_fig = COPD1_2(df)
    CVD1_4_fig = CVD1_4(df)
    CVD1_2_fig = CVD1_2(df)
    CVD1_3_fig = CVD1_3(df)
    CVD1_5_fig = CVD1_5(df)
    CVD2_0_fig = CVD2_0(df)
    AST3_1_fig = AST3_1(df)
    CVD3_1_fig = CVD3_1(df)
    CVD3_data_fig = CVD3_data(df)
    COPD5_1_fig = COPD5_1(df)
    COPD5_4_fig = COPD5_4(df)
    
    race_mor_diab_fig = race_mor_diab(df)
    race_mor_copd_fig = race_mor_copd(df)    
    race_mor_can_fig = race_mor_can(df)
    race_mor_ast_fig = race_mor_ast(df)
    gen_mor_diab_fig = gen_mor_diab(df)
    gen_mor_ast_fig = gen_mor_ast(df)
    gen_mor_can_fig = gen_mor_can(df)
    gen_mor_copd_fig = gen_mor_copd(df)
    gender_dist_cvd_mor_fig = gender_dist_cvd_mor(df)

    figsoda_comsumption = soda_comsumption(df)
    cvd_fig = cvd_treemap(df)
    immunization_heatmap_fig = immunization_heatmap(df)
    old_adult_data_fig = old_adult_data_plot(df)
    fig_oral_health_map = oral_health_map(df)

    
    return topic_counts,fig_cramers_heatmap,CKD1_0_fig,COPD1_1_fig,COPD1_2_fig,CVD1_4_fig,CVD1_2_fig,CVD1_3_fig,CVD1_5_fig,CVD2_0_fig,AST3_1_fig,CVD3_1_fig,CVD3_data_fig,COPD5_1_fig,COPD5_4_fig,\
    race_mor_diab_fig, race_mor_copd_fig, race_mor_can_fig, race_mor_ast_fig, gen_mor_diab_fig, gen_mor_ast_fig, gen_mor_can_fig, gen_mor_copd_fig, gender_dist_cvd_mor_fig,\
    figsoda_comsumption,cvd_fig,immunization_heatmap_fig,old_adult_data_fig,fig_oral_health_map
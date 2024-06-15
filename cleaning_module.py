import pandas as pd

def clean_data():
    try:
        # Loading the dataset
        df = pd.read_csv("https://storage.googleapis.com/mbcc/datasets/us_chronic_disease_indicators.csv")

        # Remove duplicate rows
        filtered_df = df.drop_duplicates()

        # Remove rows where all values are null
        filtered_df = filtered_df.dropna(how='all')

        # Check for null values
        null_counts = df.isnull().sum()

        # Check unique data value types for null datavalueunit
        null_datavalueunit_rows = df[df['datavalueunit'].isnull()]
        unique_data_value = null_datavalueunit_rows['datavaluetype'].unique()

        # Check unique locationabbr for null geolocation
        geo_null = df[df['geolocation'].isnull()]
        unique_locationabbr = geo_null['locationabbr'].unique()

        # Extract latitude and longitude from geolocation
        df[['latitude', 'longitude']] = df['geolocation'].str.extract(r'POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)')

        # Convert the extracted values to numeric
        df[['latitude', 'longitude']] = df[['latitude', 'longitude']].apply(pd.to_numeric)
    except Exception as e:
        print("Error during data cleaning:", str(e))
        return None

    return df
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('F:\SOC 23\diabaties\predictDiabetes.pkl')

pd.set_option('display.width', 500)
from flask import Flask, request, jsonify
import joblib

# Load the data
def load_data():
    data = pd.read_csv('F:\SOC 23\diabaties\diabetes.csv')
    return data

# Outlier Analysis
def outlier_thresholds (dataframe, col_name, q1=0.25, q3=0.75) :
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

   
def num_cat(df):
    num_cols = df.select_dtypes(include="number").columns.to_list()
    num_list = [col for col in df.columns if (df[col].nunique() > 10) & (col in num_cols)]

    cat_list = df.select_dtypes(include="object").columns.to_list()
    cat_list += [col for col in df.columns if (df[col].nunique() < 10) & (col not in cat_list)]

    return num_list,cat_list

def replace_with_thresholds (dataframe, variable) :
    low_limit , up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Replace zero with NaN function
def replace_zero_with_nan(dataframe, columns):
   
    dataframe.loc[:,columns] = dataframe[columns].replace(0, np.nan)

# Fill based on category function
def fill_based_cat(df, columns, based_cat, metric):
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna(df.groupby(based_cat)[col].transform(metric))
    return df


# One-hot encoder function

def create_categorical_features(df):
    # Pregnancy Category
    df['Pregnancy_Category'] = pd.cut(df['Pregnancies'], bins=[-1, 0, 1, float('inf')],
                                  labels=['Nulliparous', 'Primiparous', 'Multiparous'])

# Blood Pressure Category
    df['BloodPressure_Category'] = pd.cut(df['BloodPressure'], bins=[-1, 80, 90, float('inf')],
                                      labels=['Normal', 'Elevated', 'Hypertensive'])
    
    # Age Group
    df['Age_Group'] = pd.cut(df['Age'], bins=[-1, 30, 50, float('inf')],
                         labels=['Young Adults', 'Middle-Aged', 'Seniors'])

# BMI Category
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[-1, 18.5, 24.9, 29.9, float('inf')],
                            labels=['Underweight', 'Normal Weight', 'Overweight', 'Obese'])

# Insulin Sensitivity
    df['Insulin_Sensitivity'] = df['Glucose'] / (df['Insulin'] * df['BMI'])

# Insulin Resistance Index
    df['Insulin_Resistance_Index'] = df['Insulin'] * df['Glucose'] / df['BMI']

# Triceps Skin Fold Thickness Indicator
    df['Triceps_Skin_Fold_Indicator'] = df['SkinThickness'].apply(lambda x: 1 if 20 <= x <= 30 else 0)

    return df
def one_hot_encode(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def scale_numeric_features(dataframe, num_cols, scaler=None):

    if scaler is None:
        scaler = StandardScaler()

    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    return dataframe, scaler

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    features = [float(x) for x in request.form.values()]
   

    # Create a DataFrame from the input data
    input_data = pd.DataFrame([features], columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
    
    num_list,cat_list = num_cat(input_data)

    for col in num_list:
        replace_with_thresholds (input_data, col)


    replace_zero_with_nan(input_data, columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                                    'BMI', 'DiabetesPedigreeFunction', 'Age'])
    missing_ones = input_data.isnull().sum()[input_data.isnull().sum()>0].index

   
    input_data= create_categorical_features(input_data)

    input_data= fill_based_cat(input_data, missing_ones,"Age_Group", metric="median")

    input_data= one_hot_encode(input_data, categorical_cols=['Pregnancy_Category', 'BloodPressure_Category',
                                            'BMI_Category' , 'Age_Group'])
    
    # print(num_list)

    # input_data, scaler =scale_numeric_features(input_data,num_cols=num_list, scaler=None)

    # Make predictions
    prediction = model.predict(input_data)

    # Return the prediction to the HTML template
    return render_template('index.html', prediction_text=f'Predicted Outcome: {prediction[0]}')


if __name__ == '__main__':
    app.run(debug=True)

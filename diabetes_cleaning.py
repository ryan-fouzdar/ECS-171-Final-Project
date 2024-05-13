import pandas as pd
from pandas import read_csv

# store Training.csv in a dataframe
df = pd.read_csv('Training.csv')

# store the Testing.csv in a dataframe
df2 = pd.read_csv('Testing.csv')

# remove rows that contain 0 values in any column and store in a new dataframe
df3 = df[(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] != 0).all(1)]

# print number of rows in the cleaned dataframe
print(len(df3))

# remove rows that contain 0 values in any column other than 'Pregnancies' and store in a new dataframe
df4 = df2[(df2[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] != 0).all(1)]

# print number of rows in the cleaned dataframe
print(len(df4))

# combine the cleaned dataframes
df5 = pd.concat([df3, df4])

# store the combined dataframe in a new csv file
df5.to_csv('Diabetes.csv', index=False)
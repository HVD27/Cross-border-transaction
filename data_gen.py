import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Seed for reproducibility
np.random.seed(42)

# Define solution options
solution_options = ['Western Union', 'SEPA', 'PayPal', 'SWIFT']

# Define the size of the dataset
n_samples = 5000

# Generate features with clear patterns
data = {
    'Solution option name': np.random.choice(solution_options, n_samples),
    'Time taken to complete step': np.random.normal(50, 10, n_samples),
    'Cost of transaction fee': np.random.normal(20, 5, n_samples),
    'Cost of additional fee': np.random.normal(5, 2, n_samples),
    'Sender’s Currency': np.random.choice(['USD', 'EUR', 'JPY', 'GBP'], n_samples),
    'Sender’s country': np.random.choice(['USA', 'Germany', 'Japan', 'UK'], n_samples),
    'Recipient’s currency': np.random.choice(['USD', 'EUR', 'JPY', 'GBP'], n_samples),
    'Recipient’s country': np.random.choice(['USA', 'Germany', 'Japan', 'UK'], n_samples),
    'Transaction amount': np.random.normal(1000, 200, n_samples),
    'Settled amount': np.random.normal(995, 200, n_samples)
}

# Create a DataFrame
df_synthetic = pd.DataFrame(data)

# Introduce patterns for solution options
df_synthetic['Time taken to complete step'] = np.where(df_synthetic['Solution option name'] == 'Western Union', np.random.normal(70, 5, n_samples), df_synthetic['Time taken to complete step'])
df_synthetic['Cost of transaction fee'] = np.where(df_synthetic['Solution option name'] == 'PayPal', np.random.normal(30, 5, n_samples), df_synthetic['Cost of transaction fee'])

# Encode categorical variables
label_encoder = LabelEncoder()
df_synthetic['Sender’s Currency'] = label_encoder.fit_transform(df_synthetic['Sender’s Currency'])
df_synthetic['Sender’s country'] = label_encoder.fit_transform(df_synthetic['Sender’s country'])
df_synthetic['Recipient’s currency'] = label_encoder.fit_transform(df_synthetic['Recipient’s currency'])
df_synthetic['Recipient’s country'] = label_encoder.fit_transform(df_synthetic['Recipient’s country'])

# Feature engineering
df_synthetic['Cost Ratio'] = (df_synthetic['Cost of transaction fee'] + df_synthetic['Cost of additional fee']) / df_synthetic['Transaction amount']
df_synthetic['Time Cost Interaction'] = df_synthetic['Time taken to complete step'] * (df_synthetic['Cost of transaction fee'] + df_synthetic['Cost of additional fee'])
df_synthetic['Transaction Efficiency'] = df_synthetic['Settled amount'] / df_synthetic['Transaction amount']

# Shuffle the dataset
df_synthetic = df_synthetic.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df_synthetic.to_csv('Coherent_Cross_Border_Transactions.csv', index=False)

# Display first few rows to check the data
df_synthetic.head()

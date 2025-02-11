import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load the updated sample data from the CSV file
updated_sample_data = pd.read_csv("updated_sample_subscription_data.csv")

# Convert the relevant columns to datetime format
updated_sample_data['IncentiveDate'] = pd.to_datetime(updated_sample_data['IncentiveDate'])
updated_sample_data['SubscriptionStartDate'] = pd.to_datetime(updated_sample_data['SubscriptionStartDate'])
updated_sample_data['SubscriptionEndDate'] = pd.to_datetime(updated_sample_data['SubscriptionEndDate'])

# Prepare data for anomaly detection
features = ['Incentive', 'IncentiveDate', 'SubscriptionStartDate', 'SubscriptionEndDate']
X = updated_sample_data[features]

# Convert datetime columns to numerical values (timestamp)
X['IncentiveDate'] = X['IncentiveDate'].apply(lambda x: x.timestamp())
X['SubscriptionStartDate'] = X['SubscriptionStartDate'].apply(lambda x: x.timestamp())
X['SubscriptionEndDate'] = X['SubscriptionEndDate'].apply(lambda x: x.timestamp())

# Use Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.05)
updated_sample_data['Anomaly'] = model.fit_predict(X)
updated_sample_data['Anomaly'] = updated_sample_data['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Plot anomalies
plt.figure(figsize=(12, 6))
plt.plot(updated_sample_data.index, updated_sample_data['Incentive'], label='Incentives')
plt.scatter(updated_sample_data[updated_sample_data['Anomaly'] == 1].index, updated_sample_data[updated_sample_data['Anomaly'] == 1]['Incentive'], color='red', label='Anomalies')
plt.xlabel('Index')
plt.ylabel('Incentive')
plt.title('Anomaly Detection in Subscription Data')
plt.legend()
plt.show()

# Print the anomalies
anomalies = updated_sample_data[updated_sample_data['Anomaly'] == 1]
print("Anomalies detected:")
print(anomalies)

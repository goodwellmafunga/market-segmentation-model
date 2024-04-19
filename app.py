from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and prepare your data
try:
    data = pd.read_csv('Subscriber_Data.csv')
    data = data[['Age', 'Income', 'Spending_Score']]
    data.fillna(data.mean(), inplace=True)
except Exception as e:
    print("Error loading or processing data:", e)

# Standardizing the data
scaler = StandardScaler()
try:
    data_scaled = scaler.fit_transform(data)
except Exception as e:
    print("Error scaling data:", e)

# K-Means Clustering
try:
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_scaled)
except Exception as e:
    print("Error during K-means clustering:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        age = request.form['age']
        income = request.form['income']
        spending_score = request.form['spending_score']
        # Convert form values to float and create an array for prediction
        new_data = [float(age), float(income), float(spending_score)]
        # Standardize the new data
        new_data_scaled = scaler.transform([new_data])
        # Predict the cluster for the new data point
        cluster = kmeans.predict(new_data_scaled)[0]
        
        # Define properties for each cluster (this is a simplification)
        cluster_descriptions = {
            0: "Basic Segment: Lower income and spending score, younger age.",
            1: "Career-focused Segment: Mid-range income, higher spending score, middle-aged.",
            2: "Affluent Segment: Higher income and spending score, middle-aged to senior."
        }
        
        # Match the predicted cluster with its description
        cluster_name = cluster_descriptions.get(cluster, "Unknown Segment")
        
        # Render the results in an HTML page instead of returning JSON
        return render_template('results.html', cluster=cluster, cluster_name=cluster_name)
    except Exception as e:
        # Render an error page if something goes wrong
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

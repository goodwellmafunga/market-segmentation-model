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
        age = request.form['age']
        income = request.form['income']
        spending_score = request.form['spending_score']
        new_data = [float(age), float(income), float(spending_score)]
        new_data_scaled = scaler.transform([new_data])
        cluster = kmeans.predict(new_data_scaled)
        return jsonify({'cluster': int(cluster[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

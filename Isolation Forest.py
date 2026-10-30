import numpy as np
from sklearn.ensemble import IsolationForest
#InfluxDB

n_samples = 1000
anomaly_size = 50

def main():
    normal_traffic = np.random.normal(loc=5, scale=2, size=(n_samples, 1))
    anomaly_traffic = np.random.normal(loc=20, scale=5, size=(anomaly_size, 1))
    model = IsolationForest(contamination=0.05, random_state=100)
    model.fit(normal_traffic)
    predictions = model.predict(anomaly_traffic)
    print(predictions)

if __name__ == "__main__":
    main()

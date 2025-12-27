import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("data_sample.csv")

# Encode categorical features
encoder_type = LabelEncoder()
encoder_time = LabelEncoder()
encoder_priority = LabelEncoder()

data["incident_type"] = encoder_type.fit_transform(data["incident_type"])
data["time_of_day"] = encoder_time.fit_transform(data["time_of_day"])
data["priority"] = encoder_priority.fit_transform(data["priority"])

# Features and target
X = data[["incident_type", "description_length", "distance_km", "time_of_day"]]
y = data["priority"]

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

def predict_priority(incident_type, description_length, distance_km, time_of_day):
    it = encoder_type.transform([incident_type])[0]
    td = encoder_time.transform([time_of_day])[0]

    prediction = model.predict([[it, description_length, distance_km, td]])
    return encoder_priority.inverse_transform(prediction)[0]


# Demo run
if __name__ == "__main__":
    result = predict_priority(
        incident_type="accident",
        description_length=70,
        distance_km=3.0,
        time_of_day="night"
    )
    print("Predicted Priority:", result)
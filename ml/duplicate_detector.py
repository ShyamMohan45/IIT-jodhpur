import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

# Load existing reports
data = pd.read_csv("duplicate_data.csv")

# Convert timestamp column to datetime
data["timestamp"] = pd.to_datetime(data["timestamp"])

# Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Fit TF-IDF on existing descriptions
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["description"])

def is_duplicate(new_description, new_lat, new_lon, new_time,
                 text_threshold=0.6, distance_threshold_km=0.5, time_threshold_minutes=30):

    new_vec = vectorizer.transform([new_description])
    similarities = cosine_similarity(new_vec, tfidf_matrix)[0]

    for idx, sim in enumerate(similarities):
        if sim >= text_threshold:
            dist = haversine(
                new_lat, new_lon,
                data.loc[idx, "latitude"],
                data.loc[idx, "longitude"]
            )

            time_diff = abs(
                (new_time - data.loc[idx, "timestamp"]).total_seconds()
            ) / 60

            if dist <= distance_threshold_km and time_diff <= time_threshold_minutes:
                return True, data.loc[idx, "report_id"]

    return False, None


# Demo run
if __name__ == "__main__":
    new_report = {
        "description": "Accident near the main square with multiple vehicles",
        "latitude": 26.2390,
        "longitude": 73.0242,
        "timestamp": datetime.strptime("2025-01-01 10:08", "%Y-%m-%d %H:%M")
    }

    duplicate, report_id = is_duplicate(
        new_report["description"],
        new_report["latitude"],
        new_report["longitude"],
        new_report["timestamp"]
    )

    if duplicate:
        print(f"⚠️ Potential duplicate of report ID {report_id}")
    else:
        print("✅ New unique report")
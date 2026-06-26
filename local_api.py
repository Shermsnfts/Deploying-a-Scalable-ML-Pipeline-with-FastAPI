import requests
import json

# URL of the running API
url = "http://127.0.0.1:8001"

# 1. Send a GET using the URL
r = requests.get(url)

# Print the status code
print(f"Status Code: {r.status_code}")
# Print the welcome message
print(f"Result: {r.json()['message']}")

# Data payload for POST
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# 2. Send a POST using the data above
# We pass the dictionary to the 'json' parameter of requests.post
r_post = requests.post(f"{url}/predict", json=data)

# Print the status code
print(f"Status Code: {r_post.status_code}")
# Print the result
print(f"Result: {r_post.json()['result']}")
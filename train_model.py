print("Starting training script...")
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# 1. Load data
# Since you are likely in the root folder, use '.' for current directory
project_path = "." 
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)

# 2. Split data
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass", "education", "marital-status", "occupation", 
    "relationship", "race", "sex", "native-country"
]

# 3. Process data
X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# 4. Train model
model = train_model(X_train, y_train)

# 5. Save model, encoder, and label binarizer
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)

encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# New: Save the label binarizer (lb)
lb_path = os.path.join(project_path, "model", "lb.pkl")
save_model(lb, lb_path)

# 6. Load model and run inference
model = load_model(model_path)
preds = inference(model, X_test)

# Calculate and print metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# 7. Compute slice performance
# Clear the file first if it exists
if os.path.exists("slice_output.txt"):
    os.remove("slice_output.txt")

for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
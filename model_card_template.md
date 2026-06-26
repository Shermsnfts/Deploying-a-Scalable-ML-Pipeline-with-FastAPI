# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a Random Forest Classifier developed to predict whether an individual's salary exceeds $50,000 per year, based on the Census Income dataset.
## Intended Use
This model is intended to be used for educational purposes to practice MLOps, CI/CD, and API deployment.
## Training Data
The model was trained on the UCI Census Income dataset using an 80/20 train-test split. The preprocessing pipeline included One-Hot Encoding for categorical variables.
## Evaluation Data
The model was evaluated using a 20% hold-out test set from the original census dataset.
## Metrics
The model was evaluated using Precision, Recall, and F1-Score.
Precision: 0.7419
Recall: 0.6384
F1-Score: 0.6863
## Ethical Considerations
The model uses demographic features like race and sex, which may contain inherent historical biases. The model should be used with caution regarding protected groups.
## Caveats and Recommendations
This model is for demonstration purposes only. It should not be used for high-stakes financial decisions. Further audit for bias is recommended.
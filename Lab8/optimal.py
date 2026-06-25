import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Breast Cancer Dataset
cancer = load_breast_cancer()

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(cancer.data, cancer.target)

print("Model trained successfully")

# Sample Dataset
data = {
    'cgpa':[9.2,8.5,9.0,7.5,8.2,9.1,7.8,9.3,8.4,8.6],
    'interactiveness':['yes','no','no','no','yes','yes','yes','yes','no','yes'],
    'practical_knowledge':['verygood','good','average','average','good','good','good','verygood','good','average'],
    'communication':['good','moderate','poor','good','moderate','moderate','poor','good','good','good'],
    'job_offer':['yes','yes','no','no','yes','yes','no','yes','yes','yes']
}

df = pd.DataFrame(data)

# Encode categorical columns
encoders = {}
for col in ['interactiveness', 'practical_knowledge', 'communication', 'job_offer']:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Train Decision Tree
X = df.drop('job_offer', axis=1)
y = df['job_offer']

tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X, y)

# Plot Tree
plt.figure(figsize=(12, 6))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=encoders['job_offer'].classes_,
    filled=True
)
plt.show()

# Test Sample
test = pd.DataFrame({
    'cgpa':[6.5],
    'interactiveness':['yes'],
    'practical_knowledge':['good'],
    'communication':['good']
})

for col in ['interactiveness', 'practical_knowledge', 'communication']:
    test[col] = encoders[col].transform(test[col])

result = tree_model.predict(test)
print(
    "Predicted Job Offer for test sample:",
    encoders['job_offer'].inverse_transform(result)[0]
)

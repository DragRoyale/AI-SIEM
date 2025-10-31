import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib



# Load JSON safely
with open("brute_force_data.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Feature engineering
df['num_passwords'] = df['passwords'].apply(len)
df['avg_password_length'] = df['passwords'].apply(lambda pwds: sum(len(p) for p in pwds) / len(pwds) if pwds else 0)

# Optionally, extract IP-based features
df['ip_prefix'] = df['foreign_ip'].apply(lambda ip: '.'.join(ip.split('.')[:2]))  # First 2 octets (region-ish)
df['ip_encoded'] = df['ip_prefix'].astype('category').cat.codes  # Convert to numeric

# Assign label = 1 (all are attacks in this dataset)
df['Label'] = 1

# Define features and target
X = df[['num_passwords', 'avg_password_length', 'ip_encoded']]
y = df['Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate (note: all labels = 1, so this is just test for structure)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# joblib.dump(model, "brute_force_model.pkl")
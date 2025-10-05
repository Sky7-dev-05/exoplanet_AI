import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import json
# -------------------------------------------------
# 1. Load KOI dataset
# -------------------------------------------------
def read_csv_auto(file_path):
    """Lit un CSV en détectant automatiquement le séparateur ; ou ,"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","
        df = pd.read_csv(file_path, sep=sep)
        return df
    except Exception as e:
        raise ValueError(f"Erreur lecture CSV: {e}")

df = read_csv_auto("C:/Users/JUNIOR KPAKPA/OneDrive/Desktop/exoplanet_ML/data/cumulative1.csv")
# -------------------------------------------------
# 2. Select features
# -------------------------------------------------
features = [
    "koi_score",
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_sma",
    "koi_teq",
    "koi_model_snr"
]

feature_names_readable = [
    "Disposition Score",
    "Orbital Period [days]",
    "Impact Parameter",
    "Transit Duration [hrs]",
    "Transit Depth [ppm]",
    "Planetary Radius [Earth radii]",
    "Semi-Major Axis [au]",
    "Equilibrium Temp [K]",
    "Signal-to-Noise Ratio"
]

target = "koi_disposition"

# -------------------------------------------------
# 3. Data cleaning
# -------------------------------------------------
df = df[features + [target]].dropna()

# Encode target labels
df[target] = df[target].map({
    'FALSE POSITIVE': 0,
    'CANDIDATE': 1,
    'CONFIRMED': 2
})

# -------------------------------------------------
# 4. Split features and target
# -------------------------------------------------
X = df[features]
y = df[target]



# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=60, stratify=y
)

# -------------------------------------------------
# 5. Train ML model (XGBoost)
# -------------------------------------------------

sky_model = XGBClassifier(
    n_estimators=450,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.5,
    colsample_bytree=0.5,
    eval_metric='mlogloss',
    random_state=60,
    reg_alpha = 0.02,
    reg_lambda = 15
)


print("🚀 Training XGBoost model...")
sky_model.fit(X_train, y_train)
print("✅ Training completed!")

# -------------------------------------------------
# 6. Evaluate model & Extract metrics
# -------------------------------------------------
y_pred = sky_model.predict(X_test)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame
df_report = pd.DataFrame(report_dict).transpose()
accuracy = report_dict["accuracy"]
f1_score_macro = report_dict["macro avg"]["f1-score"]
recall_macro = report_dict["macro avg"]["recall"]
#
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)

X
score = sky_model.score(X_test, y_test)
print("Model Accuracy (score):", score)

#

metrics = {
    "accuracy": round(accuracy * 100, 2),  # en %
    "f1_score": round(f1_score_macro, 2),
    "recall": round(recall_macro, 2)
}
print(metrics)
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"],
            yticklabels=["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"])
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig("C:/Users/JUNIOR KPAKPA/OneDrive/Desktop/exoplanet_ML/graphs/confusion_matrix.jpeg", format='jpeg', dpi=300)
plt.show()

# -------------------------------------------------
# 7. FEATURE IMPORTANCE - Permutation Method (Most Reliable)
# -------------------------------------------------
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)
print("\n🔍 Computing permutation importance (this may take a moment)...")

# Calculate permutation importance
perm_importance = permutation_importance(
    sky_model, 
    X_test,
    y_test,
    n_repeats=30,  # More repeats = more stable results
    random_state=42,
    n_jobs=-1
)

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names_readable,
    'Importance': perm_importance.importances_mean,
    'Std_Dev': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\n📊 FEATURE IMPORTANCE RANKING")
print("="*60)
for idx, row in importance_df.iterrows():
    print(f"{row.name + 1}. {row['Feature']:<30} {row['Importance']:.6f} ± {row['Std_Dev']:.6f}")

# Visualization
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(importance_df)))

bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
               xerr=importance_df['Std_Dev'], 
               color=colors, alpha=0.85, 
               error_kw={'linewidth': 2, 'ecolor': 'darkred', 'capsize': 5})

ax.set_xlabel('Permutation Importance Score', fontsize=13, fontweight='bold')
ax.set_title('Most Influential Factors in Exoplanet Detection\n(Permutation Importance Method)', 
             fontsize=15, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, val, std) in enumerate(zip(bars, importance_df['Importance'], importance_df['Std_Dev'])):
    ax.text(val, bar.get_y() + bar.get_height()/2, 
            f' {val:.5f}', 
            va='center', ha='left', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("C:/Users/JUNIOR KPAKPA/OneDrive/Desktop/exoplanet_ML/graphs/permutation_importance_score.jpeg", format='jpeg', dpi=300)
plt.show()

# -------------------------------------------------
# 8. Top 3 Most Influential Factors
# -------------------------------------------------
print("\n" + "="*60)
print("🏆 TOP 3 MOST INFLUENTIAL FACTORS")
print("="*60)
top3 = importance_df.head(3)
for idx, row in top3.iterrows():
    print(f"\n{row.name + 1}. {row['Feature']}")
    print(f"   Score: {row['Importance']:.6f} (±{row['Std_Dev']:.6f})")

# -------------------------------------------------
# 9. Save model and scaler
# -------------------------------------------------
with open("C:/Users/JUNIOR KPAKPA/OneDrive/Desktop/exoplanet_ML/models/exoplanet_model.pkl", "wb") as f:
    pickle.dump(sky_model, f)

with open("C:/Users/JUNIOR KPAKPA/OneDrive/Desktop/exoplanet_ML/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save importance results
importance_df.to_csv("C:/Users/JUNIOR KPAKPA/OneDrive/Desktop/exoplanet_ML/data/feature_importance.csv", index=False)

print("\n" + "="*60)
print("✅ Model, scaler, and feature importance saved successfully!")
print("   - exoplanet_model.pkl")
print("   - scaler.pkl")
print("   - feature_importance.csv")
print("="*60)

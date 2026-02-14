import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# ✅ Put your CSV file name here (must be in the same folder)
CSV_PATH = "insurance.csv"

print("SCRIPT STARTED ✅")

df = pd.read_csv(CSV_PATH)

print("CSV LOADED ✅")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum())

# auto-detect target
target_candidates = ["charges", "premium", "cost", "insurance_cost"]
target = next((c for c in target_candidates if c in df.columns), df.columns[-1])
print("\nTarget column:", target)

# feature types
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

if target in num_cols:
    num_cols.remove(target)
if target in cat_cols:
    cat_cols.remove(target)

print("\nNumeric features:", num_cols)
print("Categorical features:", cat_cols)

# ---------- Univariate ----------
def plot_numeric(col):
    x = df[col].dropna()
    plt.figure()
    plt.hist(x, bins=30)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

    plt.figure()
    plt.boxplot(x, vert=True)
    plt.title(f"Boxplot: {col}")
    plt.ylabel(col)
    plt.show()

def plot_categorical(col):
    vc = df[col].astype("object").fillna("NaN").value_counts().head(30)
    plt.figure()
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(f"Bar chart: {col} (top 30)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# target distribution
if pd.api.types.is_numeric_dtype(df[target]):
    plt.figure()
    plt.hist(df[target].dropna(), bins=30)
    plt.title(f"Target distribution: {target}")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.show()

    plt.figure()
    plt.boxplot(df[target].dropna(), vert=True)
    plt.title(f"Target boxplot: {target}")
    plt.ylabel(target)
    plt.show()
else:
    plot_categorical(target)

for c in num_cols:
    plot_numeric(c)
for c in cat_cols:
    plot_categorical(c)

# ---------- Correlation (numeric) ----------
numeric_for_corr = [*num_cols, target] if pd.api.types.is_numeric_dtype(df[target]) else num_cols
corr = df[numeric_for_corr].corr(numeric_only=True)

plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.title("Correlation matrix (numeric)")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar()
plt.tight_layout()
plt.show()

if target in corr.columns:
    print("\nTop correlations with target:")
    print(corr[target].sort_values(ascending=False))

# ---------- Confusion matrix (bin numeric target) ----------
if pd.api.types.is_numeric_dtype(df[target]):
    y_class = pd.qcut(df[target], q=3, labels=[0, 1, 2])  # Low/Mid/High
    X = df.drop(columns=[target])

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    pre = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=3000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "Mid", "High"])
    disp.plot()
    plt.title

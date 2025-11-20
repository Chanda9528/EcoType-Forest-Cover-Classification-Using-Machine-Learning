#%%
import pandas as pd
df = pd.read_csv(r"C:\Users\admin\Downloads\forest_cover.csv")
df.head()
# %%
df.shape
df.info()
df.describe()
#%%
# Separate Features & Target Variable
TARGET = "Cover_Type"
X = df.drop(columns=[TARGET])
y = df[TARGET]

print("Target counts:\n", y.value_counts())

# %%
#  Handle Missing Values (Imputation)
from sklearn.impute import SimpleImputer
import numpy as np

numeric_cols = df.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print("Missing values handled successfully!")
print("Remaining missing:", df.isnull().sum().sum())

# %%
#  Outlier Treatment using IQR Method
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))
    return df

for col in numeric_cols:
    if col != "Cover_Type":
        df = remove_outliers_iqr(df, col)

print("Outliers treated successfully!")



# %%
#Fix Skewness in Numeric Features
skew_vals = df[numeric_cols].skew().sort_values(ascending=False)
skewed_cols = skew_vals[skew_vals > 1].index

for col in skewed_cols:
    if col != "Cover_Type":
        df[col] = np.log1p(df[col])

print("Skewness fixed for:", list(skewed_cols))


# %%
#  Encode Target Labels (Label Encoding)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Cover_Type"] = le.fit_transform(df["Cover_Type"])

print("Label Encoding Done!")
#%%
# ----- FEATURE ENGINEERING ------

# Check correlation with target
import pandas as pd

corr = df.corr()['Cover_Type'].sort_values(ascending=False)
print("Correlation with target:")
print(corr.head(15))   # Top 15 positively correlated features
print(corr.tail(15))   # Lowest correlated features

print("Feature Engineering Step Completed")
#%%

df["Distance_To_Water"] = df["Horizontal_Distance_To_Hydrology"] - df["Vertical_Distance_To_Hydrology"]

#%%
df["Distance_To_Water"].head()
#%%
# : Exploratory Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=df["Cover_Type"])
plt.title("Cover Type Class Distribution")
plt.show()
#%%
cols_for_hist = [col for col in numeric_cols if (df[col] >= 0).all()]
df[cols_for_hist].hist(figsize=(20,20), bins=30)
plt.show()

#%%
#Boxplot Visualization for Outlier Detection
plt.figure(figsize=(15,8))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=90)
plt.title("Boxplot of Numeric Features")
plt.show()

# %%
#  Correlation Analysis using Heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()



# %%
# # STEP 12: Train–Test Split
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Cover_Type"])
y = df["Cover_Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# %%
# Handle Class Imbalance using SMOTE
from imblearn.over_sampling import SMOTE
import numpy as np

print("Before SMOTE:", np.unique(y_train, return_counts=True))

sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

print("After SMOTE:", np.unique(y_train_resampled, return_counts=True))
print("Class imbalance handled successfully!")

# %%
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Train RandomForest model on balanced training data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Calculate feature importance
importances = rf_model.feature_importances_
feature_names = X.columns

# Create dataframe of feature importance
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

print(feat_imp.head(15))  # show top 15 important features

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_imp.head(15))
plt.title("Top 15 Important Features")
plt.show()


# %%
# %% 
# Model Training & Comparison

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier   
from xgboost import XGBClassifier                    
from sklearn.metrics import accuracy_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss')
}

results = {}

for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

print("Model Comparison Results:")
for model_name, accuracy in results.items():
    print(model_name, ":", accuracy)

# %%
import joblib

# Save the best-performing model ( RandomForest or XGBoost)
best_model = RandomForestClassifier(random_state=42)   #  replace with your best model
best_model.fit(X_train_resampled, y_train_resampled)

# Save model as .pkl
joblib.dump(best_model, "best_model.pkl")

print("Model saved successfully as best_model.pkl")

# %%

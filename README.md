
Experiment 7 — Preprocessing Pipeline: Impute, One-Hot Encode, Scale


AIM:

To build a ColumnTransformer pipeline in Scikit-Learn to preprocess numeric and categorical features safely and prevent data leakage.

ALGORITHM:

1. Start and import the required libraries (pandas, ColumnTransformer, Pipeline, SimpleImputer, OneHotEncoder, StandardScaler, and RandomForestClassifier).
2. Create a dataset containing both numerical features (with missing values) and categorical features, along with a target column.
3. Separate features and target variables into X and y.
4. Define the numeric pipeline: Apply SimpleImputer(strategy='median') to fill missing values, followed by StandardScaler() to normalize the features.
5. Define the categorical pipeline: Apply OneHotEncoder(handle_unknown='ignore') to convert categories into binary columns while safely handling unseen categories during testing.
6. Combine transformers: Use ColumnTransformer to route numeric columns to the numeric pipeline and categorical columns to the categorical pipeline.
7. Build the final model pipeline: Chain the preprocessor and the RandomForestClassifier into a single Pipeline object.
8. Train and verify: Fit the complete pipeline on the data (X, y) and inspect the transformed feature matrix shape.
9. Stop.

THEORY:

In machine learning, preprocessing steps like imputing missing values and scaling features must be learned strictly from the training set and then applied to the validation/test sets. If statistics (like median or mean) are computed across the entire dataset before splitting, data leakage occurs, leading to overly optimistic and invalid model evaluation.
Using Scikit-Learn's Pipeline and ColumnTransformer encapsulates all preprocessing steps into a single workflow. During cross-validation or train-test splits, the pipeline ensures transformations are fit only on the training folds and cleanly applied to unseen test data without manual intervention.

PROGRAM:

python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Create a toy DataFrame with a missing value in 'age'
df = pd.DataFrame({
    'age': [25, 35, None, 45],
    'income':,
    'city': ['X', 'Y', 'X', 'Z'],
    'target': [0, 1, 0, 1]
})

X = df[['age', 'income', 'city']]
y = df['target']

# 2. Define preprocessing for numeric features
numeric_features = ['age', 'income']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 3. Define preprocessing for categorical features
cat_features = ['city']
cat_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 4. Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', cat_transformer, cat_features)
])

# 5. Chain preprocessing and classifier into a single Pipeline
clf = Pipeline([
    ('pre', preprocessor),
    ('rf', RandomForestClassifier(random_state=42))
])

# 6. Fit the pipeline and verify output
clf.fit(X, y)

# Check transformed matrix shape (handles both sparse and dense outputs safely)
transformed_X = preprocessor.transform(X)
if hasattr(transformed_X, "toarray"):
    transformed_X = transformed_X.toarray()

print('Preprocessor output shape:', transformed_X.shape)
print('Pipeline trained successfully.')
Use code with caution.

DETAILED EXPLANATION:

1. Dataset Creation: A small DataFrame is created with numeric columns (age, income), a categorical column (city), and a binary target (target). age intentionally includes a None (NaN) value.
2. Numeric Pipeline: SimpleImputer(strategy='median') replaces the missing age with the median (35.0), and StandardScaler() scales both age and income to have zero mean and unit variance.
3. Categorical Pipeline: OneHotEncoder(handle_unknown='ignore') converts the 3 unique cities (X, Y, Z) into 3 binary columns. Setting handle_unknown='ignore' prevents errors if a new city appears in test data.
4. ColumnTransformer: Applies the numeric pipeline strictly to ['age', 'income'] and the categorical pipeline strictly to ['city'], concatenating the results side-by-side.
5. Full Pipeline: Combines preprocessor and RandomForestClassifier into one object (clf). Calling clf.fit(X, y) executes preprocessing and model training sequentially.
6. Matrix Inspection: preprocessor.transform(X) outputs a (4, 5) matrix—4 rows and 5 features (2 scaled numeric columns + 3 one-hot encoded city columns).

SAMPLE OUTPUT:

text
Preprocessor output shape: (4, 5)
Pipeline trained successfully.
Use code with caution.

RESULT:

Thus, a ColumnTransformer preprocessing pipeline was successfully implemented to impute, encode, and scale mixed data types while preventing data leakage.

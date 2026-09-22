"""Leakage-safe preprocessing for customer churn modeling."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

CATEGORICAL_FEATURES = ["Gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","TenureQ"]
NUMERIC_FEATURES = ["SeniorCitizen","Tenure_Months","MonthlyCharges","TotalCharges"]
TARGET_COLUMN = "Churn"

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size=test_size
        self.random_state=random_state
        self.numeric_features=list(NUMERIC_FEATURES)
        self.categorical_features=list(CATEGORICAL_FEATURES)
        self.feature_columns=self.numeric_features+self.categorical_features
        self.logistic_preprocessor=None
        self.tree_preprocessor=None

    def clean_data(self, df):
        out=df.copy()
        if "TotalCharges" in out:
            out["TotalCharges"]=pd.to_numeric(out["TotalCharges"], errors="coerce")
        for c in self.numeric_features:
            if c in out and out[c].isna().any():
                out[c]=out[c].fillna(out[c].median())
        for c in self.categorical_features:
            if c in out and out[c].isna().any():
                mode=out[c].mode(dropna=True)
                out[c]=out[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
        if TARGET_COLUMN in out:
            out[TARGET_COLUMN]=pd.to_numeric(out[TARGET_COLUMN], errors="raise").astype(int)
        return out

    def split_raw_data(self, df):
        cleaned=self.clean_data(df)
        X=cleaned[[c for c in self.feature_columns if c in cleaned.columns]].copy()
        y=cleaned[TARGET_COLUMN].copy()
        return train_test_split(X,y,test_size=self.test_size,random_state=self.random_state,stratify=y)

    def _make_transformer(self, scale_numeric):
        num=[("imputer",SimpleImputer(strategy="median"))]
        if scale_numeric:
            num.append(("scaler",StandardScaler()))
        numeric=Pipeline(num)
        categorical=Pipeline([
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
        ])
        return ColumnTransformer([
            ("num",numeric,self.numeric_features),
            ("cat",categorical,self.categorical_features)
        ],remainder="drop")

    def fit_transformers(self, X_train):
        self.logistic_preprocessor=self._make_transformer(True)
        self.tree_preprocessor=self._make_transformer(False)
        self.logistic_preprocessor.fit(X_train)
        self.tree_preprocessor.fit(X_train)

    def transform(self, X, model_type="tree"):
        transformer=self.logistic_preprocessor if model_type=="logistic" else self.tree_preprocessor
        if transformer is None:
            raise ValueError("Transformers must be fitted on training data first.")
        return transformer.transform(X)

    def get_feature_names(self, model_type="tree"):
        transformer=self.logistic_preprocessor if model_type=="logistic" else self.tree_preprocessor
        if transformer is None:
            raise ValueError("Transformer has not been fitted.")
        return list(transformer.get_feature_names_out())

    def preprocess_full(self, df):
        X_train,X_test,y_train,y_test=self.split_raw_data(df)
        self.fit_transformers(X_train)
        names=self.get_feature_names("tree")
        return (
            pd.DataFrame(self.transform(X_train,"tree"),index=X_train.index,columns=names),
            pd.DataFrame(self.transform(X_test,"tree"),index=X_test.index,columns=names),
            y_train,y_test
        )

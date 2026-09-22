"""Model comparison and evaluation for churn prediction."""
from pathlib import Path
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix

class ChurnClassifier:
    def __init__(self,preprocessor=None,random_state=42):
        self.preprocessor=preprocessor
        self.random_state=random_state
        self.models={}
        self.metrics={}
        self.best_params={}
        self.model=None
        self.model_name="random_forest"
        self.feature_importances={}
        self._is_trained=False

    def fit(self,X_train,y_train,use_smote=False,tune_params=True):
        if self.preprocessor is None:
            raise ValueError("A DataPreprocessor is required.")
        self.preprocessor.fit_transformers(X_train)
        Xlog=self.preprocessor.transform(X_train,"logistic")
        Xtree=self.preprocessor.transform(X_train,"tree")
        dummy=DummyClassifier(strategy="most_frequent")
        logistic=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=self.random_state)
        rf=RandomForestClassifier(n_estimators=300,class_weight="balanced_subsample",random_state=self.random_state,n_jobs=-1)
        dummy.fit(Xlog,y_train)
        logistic.fit(Xlog,y_train)
        if tune_params:
            grid=GridSearchCV(
                rf,
                {"max_depth":[None,8],"min_samples_leaf":[1,3],"max_features":["sqrt"]},
                scoring="roc_auc",
                cv=StratifiedKFold(3,shuffle=True,random_state=self.random_state),
                n_jobs=-1
            )
            grid.fit(Xtree,y_train)
            rf=grid.best_estimator_
            self.best_params=grid.best_params_
        else:
            rf.fit(Xtree,y_train)
            self.best_params=rf.get_params()
        self.models={"majority_baseline":dummy,"logistic_regression":logistic,"random_forest":rf}
        self.model=rf
        self.feature_importances=dict(zip(self.preprocessor.get_feature_names("tree"),rf.feature_importances_))
        self._is_trained=True
        return self

    def _predict_proba(self,name,X):
        Xt=self.preprocessor.transform(X,"logistic" if name=="logistic_regression" else "tree")
        return self.models[name].predict_proba(Xt)[:,1]

    def evaluate(self,X_test,y_test):
        if not self._is_trained:
            raise ValueError("Models must be trained before evaluation.")
        results={}
        for name in self.models:
            p=self._predict_proba(name,X_test)
            pred=(p>=0.5).astype(int)
            tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
            results[name]={
                "accuracy":float(accuracy_score(y_test,pred)),
                "precision":float(precision_score(y_test,pred,zero_division=0)),
                "recall":float(recall_score(y_test,pred,zero_division=0)),
                "f1_score":float(f1_score(y_test,pred,zero_division=0)),
                "roc_auc":float(roc_auc_score(y_test,p)),
                "pr_auc":float(average_precision_score(y_test,p)),
                "true_negatives":int(tn),"false_positives":int(fp),
                "false_negatives":int(fn),"true_positives":int(tp)
            }
        self.metrics=results
        return results

    def predict_proba(self,X,model_name=None):
        if not self._is_trained:
            raise ValueError("Models must be trained before prediction.")
        return self._predict_proba(model_name or self.model_name,X)

    def predict(self,X,threshold=0.5,model_name=None):
        return (self.predict_proba(X,model_name)>=threshold).astype(int)

    def get_model(self,name):
        return self.models[name]

    def save_model(self,filepath):
        if not self._is_trained:
            raise ValueError("No trained model to save.")
        Path(filepath).parent.mkdir(parents=True,exist_ok=True)
        joblib.dump({
            "model":self.model,
            "models":self.models,
            "best_params":self.best_params,
            "feature_importances":self.feature_importances,
            "model_name":self.model_name
        },filepath)

    def load_model(self,filepath):
        payload=joblib.load(filepath)
        self.model=payload["model"]
        self.models=payload["models"]
        self.best_params=payload.get("best_params",{})
        self.feature_importances=payload.get("feature_importances",{})
        self.model_name=payload.get("model_name","random_forest")
        self._is_trained=True
        return self

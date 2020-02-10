import xgboost as xgb

class XGBoost:
    def __init__(self):
        self.xgb = xgb.XGBClassifier(n_estimators=500, max_depth=5, base_score=0.5,
                                objective='binary:logistic', random_state=42)
        self.title = 'XGBoost'

    def fit(self, X_train, y_train):
        self.xgb.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.xgb.predict(X_test)
        return y_pred

    def get_model(self):
        return self.xgb

    def get_title(self):
        return self.title
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self):
        self.rfmodel = RandomForestClassifier(
            n_estimators=80, max_depth=50  # class_weight= {0:.1, 1:.9},
        )

    def fit(self, X_train, y_train):
        self.rfmodel.fit(X_train, y_train)

    def predict(self, X_test, y_test):
        self.rfmodel.predict(X_test, y_test)

    def get_model(self):
        return self.rfmodel

from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self):
        self.rf = RandomForestClassifier(
            n_estimators=80, max_depth=50  # class_weight= {0:.1, 1:.9},
        )
        self.title = 'Random Forest'

    def fit(self, X_train, y_train):
        self.rf.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.rf.predict(X_test)
        return y_pred

    def get_model(self):
        return self.rf

    def get_title(self):
        return self.title

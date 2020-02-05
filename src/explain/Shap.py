import shap


class Shap:
    def __init__(self, model_type, model, data=None, category=None):
        shap.initjs()
        if model_type == 'Neural Network':
            self.explainer = shap.DeepExplainer(model, data)
        else:
            self.explainer = shap.TreeExplainer(model)
        self.category = category
        self.shap_values = None

    def explain(self, data):
        self.shap_values = self.explainer.shap_values(data)
        self.expected_value = self.explainer.expected_value
        if self.category is not None:
            self.shap_values = self.shap_values[self.category]
            self.expected_value = self.explainer.expected_value[self.category]

    def plot_summary(self, data):
        shap.summary_plot(self.shap_values, data, plot_type="bar")

    def plot_force(self, data, index):
        shap.force_plot(self.expected_value,
                        self.shap_values[index, :],
                        data,
                        matplotlib=True,
                        link='logit')

    def plot_forces(self, data, index):
        shap.force_plot(self.expected_value,
                        self.shap_values[:index, :],
                        data,
                        matplotlib=True,
                        link='logit')

    def get_shap_values(self):
        return self.shap_values

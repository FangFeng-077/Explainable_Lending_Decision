import shap

class Shap:
    def __init__(self, model_type, model, background=None, category=None):
        shap.initjs()
        self.model_type = model_type
        if self.model_type == 'Neural Network':
            self.explainer = shap.DeepExplainer(model, [background])
        else:
            self.explainer = shap.TreeExplainer(model)
        self.category = category
        self.shap_values = None
        self.expected_value = None

    def explain(self, data):
        self.shap_values = self.explainer.shap_values(data)
        self.expected_value = self.explainer.expected_value

        # show only one category or not, 1 means paid
        if self.model_type == 'Random Forest':
            self.shap_values = self.shap_values[1]
            self.expected_value = self.explainer.expected_value[1]

        # neural network
        if self.model_type == 'Neural Network':
            self.shap_values = self.shap_values[0]
            self.expected_value = self.explainer.expected_value[0]

    def plot_summary(self, data):
        shap.summary_plot(self.shap_values, data, plot_type="bar")

    def plot_force(self, data, index):
        if self.expected_value is None:
            return
        if self.shap_values is None:
            return
        shap.force_plot(self.expected_value,
                        self.shap_values[index, :],
                        data,
                        link='logit',
                        matplotlib=True)

    def plot_forces(self, data, index):
        if self.expected_value is None:
            return
        if self.shap_values is None:
            return
        shap.force_plot(self.expected_value,
                        self.shap_values[:index, :],
                        data,
                        link='logit')

    def plot_image(self, data, index):
        if self.expected_value is None:
            return
        if self.shap_values is None:
            return
        shap.image_plot(self.shap_values[index, :], data)


    def get_shap_values(self):
        return self.shap_values

    def get_expected_value(self):
        return self.expected_value

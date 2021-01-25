from data_exploratory import DataExploratory
from model_optimizer import ModelOptimizer
from pytorch_regression_model import PytorchRegression


class Main:
    def __init__(self, train, test, id_column, y_column_name, num_of_features_to_select):
        self.exploratory = DataExploratory(train, test)
        self.model = ModelOptimizer(train, test, id_column, y_column_name, num_of_features_to_select)
        self.pytorch = PytorchRegression(train, test, id_column, y_column_name, num_of_features_to_select)

    def check_missing_values(self):
        return self.exploratory.get_missing_values()

    def check_numerical_features(self):
        return self.exploratory.get_numerical_features()

    def check_categorical_features(self):
        return self.exploratory.get_categorical_features()

    def cross_validation_random_forest_regressor(self):
        return self.model.perform_cross_validation_random_forest()

    def cross_validation_random_extra_tree_regressor(self):
        return self.model.perform_cross_validation_extra_tree_regressor()

    def cross_validation_random_xgb_regressor(self):
        return self.model.perform_cross_validation_xgb()

    def optimize_and_train_model(self, predicted_column_name):
        return self.model.optimize_train_model(predicted_column_name)

    def train_with_pytorch(self):
        return self.pytorch.train_model_with_torch()

    def prediction_pytorch(self):
        return self.pytorch.model_prediction_pytorch()






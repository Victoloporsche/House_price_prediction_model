from data_exploratory import DataExploratory
from model_optimizer import ModelOptimizer


class Main:
    def __init__(self, train, test, id_column, y_column_name):
        self.exploratory = DataExploratory(train, test)
        self.model = ModelOptimizer(train, test, id_column, y_column_name)

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






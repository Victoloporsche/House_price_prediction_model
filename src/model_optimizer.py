from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
import pandas as pd
from preprocessed_data import PreprocessedData
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

class ModelOptimizer:
    def __init__(self, train, test, id_column, y_column_name, num_of_features_to_select):
        self.train = train
        self.number_of_train = train.shape[0]
        self.y_column_name = y_column_name
        self.id_column = id_column
        self.test = test
        self.data = pd.concat([train, test], ignore_index=True)
        self.processed_data = PreprocessedData(train, test, id_column, y_column_name)
        self.data = self.processed_data.preprocess_my_data(num_of_features_to_select)
        self.train_data = self.data[:self.number_of_train]
        self.ytrain = self.train_data[[self.y_column_name]]
        self.xtrain = self.train_data.drop([self.id_column, self.y_column_name], axis=1)
        self.test_data = self.data[self.number_of_train:]
        self.xtest = self.test_data.drop([self.id_column, self.y_column_name], axis=1)

    def perform_cross_validation_random_forest(self, kfold=4):
        cross_validate = cross_val_score(RandomForestRegressor(), self.xtrain,  self.ytrain, cv=kfold)
        mean_cross_validation_score = cross_validate.mean()
        return mean_cross_validation_score

    def perform_cross_validation_extra_tree_regressor(self, kfold=4):
        cross_validate = cross_val_score(ExtraTreesRegressor(), self.xtrain,  self.ytrain, cv=kfold)
        mean_cross_validation_score = cross_validate.mean()
        return mean_cross_validation_score

    def perform_cross_validation_xgb(self, kfold=4):
        cross_validate = cross_val_score(xgb.XGBRegressor(), self.xtrain,  self.ytrain, cv=kfold)
        mean_cross_validation_score = cross_validate.mean()
        return mean_cross_validation_score

    def optimize_train_model(self, predicted_column_name):
        param_dist = {
            'max_depth': [5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
            'n_estimators': [50, 100, 150, 200]
        }

        xg_random = RandomizedSearchCV(xgb.XGBRegressor(), param_distributions=param_dist)

        fit_model = xg_random.fit(self.xtrain,  self.ytrain)
        predict_model = fit_model.predict(self.xtest)
        y_predict = pd.DataFrame(data=predict_model, columns=[predicted_column_name])
        return y_predict, fit_model





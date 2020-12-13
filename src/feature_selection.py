from sklearn.ensemble import ExtraTreesRegressor
from feature_engineering import FeatureEngineering
import pandas as pd
import matplotlib.pyplot as plt

class FeatureSelection:

    def __init__(self, train , test, id_column, y_column_name):
        self.train = train
        self.number_of_train = train.shape[0]
        self.y_column_name = y_column_name
        self.id_column = id_column
        self.test = test
        self.data = pd.concat([train, test], ignore_index=True)
        self.feature_engineering = FeatureEngineering(train, test, id_column, y_column_name)
        self.y = self.data[[self.id_column, self.y_column_name]]

    def preprocess_my_data(self):
        self.data = self.feature_engineering.fill_na_categorical()
        self.data = self.feature_engineering.fill_na_numerical()
        self.data = self.feature_engineering.input_rare_categorical()
        self.data = self.feature_engineering.label_encoder()
        self.data = self.feature_engineering.get_scale_features()
        return self.data

    def perform_feature_selection(self, num_of_features_to_select):
        data = self.preprocess_my_data()
        train_data = data[:self.number_of_train]
        ytrain = train_data[[self.y_column_name]]
        xtrain = train_data.drop([self.id_column, self.y_column_name], axis=1)
        feature_sel_model = ExtraTreesRegressor().fit(xtrain, ytrain)
        feat_importances = pd.Series(feature_sel_model.feature_importances_, index=xtrain.columns)
        selected_features = feat_importances.nlargest(num_of_features_to_select)
        selected_features_df = selected_features.to_frame()
        selected_features_list = selected_features_df.index.tolist()
        data = self.data[selected_features_list]
        self.data = pd.concat([self.y, data], axis=1)
        return self.data



import pandas as pd
from feature_engineering import FeatureEngineering
from feature_selection import FeatureSelection

class PreprocessedData:
    def __init__(self, train , test, id_column, y_column_name):
        self.train = train
        self.test = test
        self.y_column_name = y_column_name
        self.id_column = id_column
        self.data = pd.concat([train, test], ignore_index=True)
        self.feature_engineering = FeatureEngineering(train, test, id_column, y_column_name)
        self.feature_selection = FeatureSelection(train, test, id_column, y_column_name)

    def preprocess_my_data(self):
        self.data = self.feature_engineering.fill_na_categorical()
        self.data = self.feature_engineering.fill_na_numerical()
        self.data = self.feature_engineering.input_rare_categorical()
        self.data = self.feature_engineering.label_encoder()
        self.data = self.feature_engineering.get_scale_features()
        self.data = self.feature_selection.perform_extra_regressor_feature_selection()
        return self.data



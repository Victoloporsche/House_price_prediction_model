from data_exploratory import DataExploratory
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineering:
    def __init__(self):
        print("Feature Engineering is starting...")

        self.exploratory = DataExploratory()

    def fillna_categorical(self, data):
        missing_categorical_features = self.exploratory._get_missing_categorical(data)
        for feature in missing_categorical_features:
            data[feature] = data[feature].fillna("Missing")
        return data

    def fillna_numerical(self, data):
        missing_numerical_feature = self.exploratory._get_missing_numerical(data)
        for feature in missing_numerical_feature:
            data[feature] = data[feature].fillna(data[feature].mean())
        return data

    def input_rare_categorical(self, data):
        categorical_features = self.exploratory._get_categorical_variables(data)
        for feature in categorical_features:
            temp = data.groupby(feature)['SalePrice'].count() / len(data)
            temp_df = temp[temp > 0.01].index
            data[feature] = np.where(data[feature].isin(temp_df), data[feature], 'Rare_var')
        return data

    def label_encoder(self, data):
        labelEncoder = LabelEncoder()
        categorical_features = self.exploratory._get_categorical_variables(data)
        for feature in categorical_features:
            data[feature] = labelEncoder.fit_transform(data[feature])
        return data

    def _get_mappings(self, data):
        categorical_features = self.exploratory._get_categorical_variables(data)
        labelEncoder = LabelEncoder()
        mapping_dict = {}
        for feature in categorical_features:
            data[feature] = labelEncoder.fit_transform(data[feature])
            cat_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
            mapping_dict[feature] = cat_mapping
        return mapping_dict

    def _save_mappings(self, data):
        encoded_dict = self._get_mappings(data)

        with open('../output/dict_house_price.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in encoded_dict.items():
                writer.writerow([key, value])

    def _get_scale_features(self,data):
        data = self.fillna_categorical(data)
        data = self.label_encoder(data)
        scaling_feature = [feature for feature in data.columns if feature not in ['Id', 'SalePrice']]
        scaling_features_data = data[scaling_feature]
        scaler = MinMaxScaler()
        scaler.fit(data[scaling_feature])
        scaled_data = scaler.transform(data[scaling_feature])
        return scaled_data

    def _get_scaled_with_dependent_features(self, data):
        scaled_data = self._get_scale_features(data)
        scaling_feature = self.scaling_feature(data)
        full_data = pd.concat([data[['Id', 'SalePrice']].reset_index(drop=True),
                          pd.DataFrame(scaled_data, columns=scaling_feature)] , axis=1)
        return full_data

    def scaling_feature(self,data):
        data = self.fillna_categorical(data)
        data = self.label_encoder(data)
        scaling_feature = [feature for feature in data.columns if feature not in ['Id', 'SalePrice']]
        return scaling_feature





from data_exploratory import DataExploratory
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineering:
    def __init__(self, train, test, id_column, y_column_name):
        self.train = train
        self.test = test
        self.y_column_name = y_column_name
        self.id_column = id_column
        self.data = pd.concat([train, test], ignore_index=True)
        self.exploratory = DataExploratory(train, test)


    def fill_na_categorical(self):
        cat_features_nan = [feature for feature in self.data.columns if
                        self.data[feature].isnull().sum() > 1 and self.data[feature].dtypes == 'O']
        self.data[cat_features_nan] = self.data[cat_features_nan].fillna("Missing")
        return self.data

    def fill_na_numerical(self):
        numerical_with_nan = [feature for feature in self.data.columns if
                              self.data[feature].isnull().sum() > 1 and self.data[feature].dtypes != 'O']
        self.data[numerical_with_nan] = self.data[numerical_with_nan].fillna(self.data[numerical_with_nan].mean())
        return self.data

    def input_rare_categorical(self):
        categorical_features = self.exploratory.get_categorical_features()
        for feature in categorical_features:
            temp = self.data.groupby(feature)[self.y_column_name].count() / len(self.data)
            temp_df = temp[temp > 0.01].index
            self.data[feature] = np.where(self.data[feature].isin(temp_df), self.data[feature], 'Rare_var')
        return self.data

    def label_encoder(self):
        labelEncoder = LabelEncoder()
        categorical_features = self.exploratory.get_categorical_features()
        mapping_dict = {}
        for feature in categorical_features:
            self.data[feature] = labelEncoder.fit_transform(self.data[feature])
            cat_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
            mapping_dict[feature] = cat_mapping

        with open('../output/dict_house_price.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in mapping_dict.items():
                writer.writerow([key, value])
        return self.data

    def get_scale_features(self):
        scaler = MinMaxScaler()
        scaling_feature = [feature for feature in self.data.columns if feature not in [self.id_column,
                                                                                       self.y_column_name]]
        scaling_features_data = self.data[scaling_feature]
        scale_fit = scaler.fit(scaling_features_data)
        scale_transform = scaler.transform(scaling_features_data)

        data = pd.concat([self.data[[self.id_column, self.y_column_name]].reset_index(drop=True),
                          pd.DataFrame(scaler.transform(self.data[scaling_feature]), columns=scaling_feature)],
                         axis=1)
        self.data = data
        return self.data





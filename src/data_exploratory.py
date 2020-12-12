import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataExploratory:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.data = pd.concat([train, test], ignore_index=True)

    def get_missing_values(self):
        features_with_missing_values = [features for features in self.data.columns if
                                        self.data[features].isnull().sum()>0]
        for features in features_with_missing_values:
            #print(features, np.round(self.data[features].isnull().sum(), 4), "missing Values")
            na_features_data = self.data[features_with_missing_values]
        return na_features_data

    def get_numerical_features(self):
        numerical_features = [features for features in self.data.columns if self.data[features].dtype != 'O']
        numerical_features_data = self.data[numerical_features]
        return numerical_features_data

    def get_year_features(self):
        numerical_data = self.get_numerical_features()
        year_feature = [feature for feature in numerical_data if 'Yr' in feature or
                        'Year' in feature]

        year_features_data = self.data[year_feature]
        return year_features_data

    def obtain_discrete_features(self):
        numerical_features = self.get_numerical_features()
        year_features = self.get_year_features()
        discrete_features = [feature for feature in numerical_features if
                             len(self.data[feature].unique())<25 and feature not in year_features]
        discrete_features_data = self.data[discrete_features]
        return discrete_features_data

    def obtain_continous_features(self):
        numerical_features = self.get_numerical_features()
        discrete_features = self.obtain_discrete_features()
        year_feature = self.get_year_features()
        continous_feature = [feature for feature in numerical_features if feature
                             not in discrete_features + year_feature]
        continous_features_data = self.data[continous_feature]
        return continous_features_data

    def get_outliers_discrete(self):
        discrete_features = self.obtain_discrete_features()
        for feature in discrete_features:
            self.data[feature] = np.log(self.data[feature])
            self.data.boxplot(column=feature)
            plt.ylabel(feature)
            plt.title(feature)
            plt.show()

    def get_outliers_continous(self):
        continous_features = self.obtain_continous_features()
        for feature in continous_features:
            self.data[feature] = np.log(self.data[feature])
            self.data.boxplot(column=feature)
            plt.ylabel(feature)
            plt.title(feature)
            plt.show()

    def get_categorical_features(self):
        categorical_features = [feature for feature in self.data.columns if
                                self.data[feature].dtypes == 'O']
        categorical_features_data = self.data[categorical_features]
        return categorical_features_data

    def get_missing_categorical(self):
        nan_categorical = [feature for feature in self.data.columns if
                           self.data[feature].isnull().sum() > 0 and self.data[feature].dtypes == 'O']
        nan_categorical_data = self.data[nan_categorical]
        return nan_categorical_data

    def get_missing_numerical(self):
        missing_numerical = [feature for feature in self.data.columns if
                             self.data[feature].isnull().sum() > 0 and self.data[feature].dtypes != 'O']

        nan_numerical_data = self.data[missing_numerical]
        return nan_numerical_data







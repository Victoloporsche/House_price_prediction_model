import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataExploratory:
    def __init__(self):
        print("Explortory Analysis of this data is starting...")

    def _get_missing_values(self, data):
        features_with_missing_values = [features for features in data.columns if data[features].isnull().sum()>0]
        for features in features_with_missing_values:
            print(features, np.round(data[features].isnull().sum(), 4), "missing Values")
            na_features_data = data[features_with_missing_values]
            return na_features_data
        else:
            print("There are no missing values in this dataset")

    def _get_missing_values_with_dependent(self, data):
        na_features = self._get_missing_values(data)
        for feature in na_features:
            data[feature] = np.where(data[feature].isnull(), 1, 0)
            data.groupby(feature)['SalePrice'].median().plot.bar()
            plt.title(feature)
            plt.show()

    def _get_numerical_features(self, data):
        numerical_features = [features for features in data.columns if data[features].dtype != 'O']
        print('The number of numerical features are:', len(numerical_features))
        numerical_features_data = data[numerical_features]
        return numerical_features_data

    def _get_year_features(self, data):
        numerical_data = self._get_numerical_features(data)
        year_feature = [feature for feature in numerical_data if 'Yr' in feature or
                        'Year' in feature]
        print('The number of year features are:', len(year_feature))
        year_features_data = data[year_feature]
        return year_features_data

    def plot_year_with_dependent_feature(self, data):
        year_feature = self._get_year_features(data)
        for feature in year_feature:
            data.groupby(feature)['SalePrice'].median().plot()
            plt.title(feature)
            plt.show()

    def _obtain_discrete_variables(self, data):
        numerical_features = self._get_numerical_features(data)
        year_features = self._get_year_features(data)
        discrete_features = [feature for feature in numerical_features if
                             len(data[feature].unique())<25 and feature not in
                             year_features + ['Id']]
        print('Discrete features count: {}'.format(len(discrete_features)))
        discrete_features_data = data[discrete_features]
        return discrete_features_data

    def plot_discrete_dependent_feature(self, data):
        discrete_features = self._obtain_discrete_variables(data)
        for feature in discrete_features:
            data.groupby(feature)['SalePrice'].median().plot.bar()
            plt.xlabel(feature)
            plt.ylabel('SalePrice')
            plt.title(feature)
            plt.show()

    def _obtain_continous_features(self, data):
        numerical_features = self._get_numerical_features(data)
        discrete_features = self._obtain_discrete_variables(data)
        year_feature = self._get_year_features(data)
        continous_feature = [feature for feature in numerical_features if feature
                             not in discrete_features + year_feature + ['Id']]
        print("continous feature count {}".format(len(continous_feature)))
        continous_features_data = data[continous_feature]
        return continous_features_data

    def plot_continous_and_dependent_feature(self, data):
        continous_feature = self._obtain_continous_features(data)
        for feature in continous_feature:
            data.groupby(feature)['SalePrice'].mean().hist(bins=25)
            plt.xlabel(feature)
            plt.ylabel('SalePrice')
            plt.title(feature)
            plt.show()

    def _get_outliers_discrete(self, data):
        discrete_features = self._obtain_discrete_variables(data)
        for feature in discrete_features:
                data[feature]= np.log(data[feature])
                data.boxplot(column=feature)
                plt.ylabel(feature)
                plt.title(feature)
                plt.show()

    def _get_outliers_continous(self, data):
        continous_features = self._obtain_continous_features(data)
        for feature in continous_features:
                data[feature] = np.log(data[feature])
                data.boxplot(column=feature)
                plt.ylabel(feature)
                plt.title(feature)
                plt.show()

    def _get_categorical_variables(self, data):
        categorical_features = [feature for feature in data.columns if
                                data[feature].dtypes=='O']
        print('Number of categorical fetures: {}'.format(len(categorical_features)))
        categorical_features_data = data[categorical_features]
        return categorical_features_data

    def unique_values_categorical(self, data):
        categorical_features = self._get_categorical_variables(data)
        for feature in categorical_features:
            print('The feature {} and the number of unique categories are {}'
                  .format(feature,len(data[feature].unique())))

    def plot_categorical_dependent_feature(self, data):
        categorical_features = self._get_categorical_variables(data)
        for feature in categorical_features:
            data.groupby(feature)['SalePrice'].median().plot.bar()
            plt.xlabel(feature)
            plt.ylabel('SalePrice')
            plt.title(feature)
            plt.show()

    def _get_missing_categorical(self, data):
        missing_categorical = [feature for feature in data.columns if
                        data[feature].isnull().sum() > 0 and data[feature].dtypes == 'O']
        for feature in missing_categorical:
            print('{}: {} missing values'.format(feature, np.round(data[feature].isnull().sum(), 4)))
        na_categorical_features_data = data[missing_categorical]
        return na_categorical_features_data

    def _get_missing_numerical(self,data):
        missing_numerical = [feature for feature in data.columns if
                             data[feature].isnull().sum()>0 and data[feature].dtypes!='O']
        for feature in missing_numerical:
            print('{}: {} missing value'.format(feature, np.round(data[feature].isnull().sum(), 4)))
            na_numerical_features_data = data[missing_numerical]
        return na_numerical_features_data














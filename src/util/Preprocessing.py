import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np


class Preprocessor:
    def __init__(self):
        pass

    def normalize(self, df, columns):
        min_max_scaler = preprocessing.MinMaxScaler()
        df[columns] = min_max_scaler.fit_transform(df[columns])
        return df

    def select(self, input_df, columns):
        return input_df[columns]

    def filter(self, input_df, col, valid_status):
        return input_df[input_df[col].isin(valid_status)]

    def transform(self):
        pass

    def under_sample(self, input_df, ratio=1.0, random_state=3):
        """Undersamples the majority class to reach a ratio by default
            equal to 1 between the majority and minority classes"""
        count_class_0, count_class_1 = input_df["Status"].value_counts()
        df_class_0 = input_df[input_df["Status"] == "paid"]
        df_class_1 = input_df[input_df["Status"] == "defaulted"]
        df_class_0_under = df_class_0.sample(
            int(ratio * count_class_1), random_state=random_state
        )
        df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        return df_train_under

    def over_sample(self, input_df, ratio=1.0, random_state=3):
        """Oversamples the minority class to reach a ratio by default
            equal to 1 between the majority and mionority classes"""
        count_class_0, count_class_1 = input_df["Status"].value_counts()
        df_class_0 = input_df[input_df["Status"] == "paid"]
        df_class_1 = input_df[input_df["Status"] == "defaulted"]
        df_class_1_over = df_class_1.sample(
            int(ratio * count_class_0), replace=True, random_state=random_state
        )
        df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)
        return df_train_over

    def split(self, input_df, test_size=0.3, random_state=3):
        train, test = train_test_split(input_df, test_size=test_size, random_state=random_state)
        return train, test
### no used cases

    def ohe_encode(self, input_df, categorical_columns, ordinal_columns):

        ohe = preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False)
        X = np.transpose(ohe.fit_transform(input_df[categorical_columns]))

        for c in ordinal_columns:
            X = np.vstack([X, input_df[c]])
        X = np.transpose(X)

        features = ohe.get_feature_names(categorical_columns).tolist()
        for c in ordinal_columns:
            features.append(c)
        X_df = pd.DataFrame(X, columns=features)

        return X_df

    def label_encode(self, df, categorical_columns):
        for cal in categorical_columns:
            df[cal] = df[cal].astype('category')
        cat_cols = df.select_dtypes(['category']).columns
        df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)
        return df

    def write_to_csv(self, input_df, path):
        input_df.to_csv(path)

    def transformFundedTime(self, df):
        # A new feature "Funded Time" gives the exact time when the loan was funded.
        df["Funded Time"] = df.apply(lambda row: row['Funded Date.year'] + 0.833 * row['Funded Date.month'], axis=1)
        return df

    def transformCountryCurrency(self, df):
        df['Country Currency'] = df.apply(lambda row: row.Country + '_' + row.Currency, axis=1)
        return df

    def transformStatus(self, df):
        df['Status'] = pd.get_dummies(df["Status"], columns=["Status"])["defaulted"]
        return df

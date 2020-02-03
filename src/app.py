import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from util.Preprocessing import Preprocessor
from explain.Shap import Shap
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import os

st.title('Explainable Lending Decision')


DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz')


@st.cache
def load_data(path, nrows=None):
    data = pd.read_csv(path, nrows=nrows)
    return data


def preprocessing(df, config):
    print('start preprocessing')
    useful_columns = [
        "Funded Time",
        "Loan Amount",
        "Loan Term",
        "Town",
        "Country",
        "Sector",
        "Activity",
        "Partner ID",
        "Country Currency",
        "Status"
    ]
    categorical_columns = [
        "Country",
        "Sector",
        "Activity",
        "Partner ID",
        "Country Currency",
        "Town"
    ]
    ordinal_columns = ["Loan Amount", "Loan Term", "Funded Time", "Status"]
    normalize_columns = ["Loan Amount", "Loan Term", "Funded Time"]

    preprocessor = Preprocessor()

    # clean df Status in ['paid', 'defauted']
    df = preprocessor.filter(df, 'Status', ['paid', 'defaulted'])

    # sampling
    if config.sample == 'undersample':
        df = preprocessor.under_sample(df)
    elif config.sample == 'oversample':
        df = preprocessor.over_sample(df)


    # transform feature
    df = preprocessor.transformFundedTime(df)
    df = preprocessor.transformStatus(df)
    df = preprocessor.transformCountryCurrency(df)

    # select useful columns
    df = preprocessor.select(df, useful_columns)

    df_disp = df.copy()
    # visualize processed data
    st.subheader('Sampled data')
    st.write(df)

    # normalize
    df = preprocessor.normalize(df, normalize_columns)

    # encode category data
    if config.encode == 'ohe':
        # ohe encode
        df = preprocessor.ohe_encode(df, categorical_columns, ordinal_columns)
    elif config.encode == 'label':
        # label encode
        df = preprocessor.label_encode(df, categorical_columns)

    return df, df_disp


def build_model(solver, X_train, y_train,):
    if solver == 'Random Forest':
        print("training random forest")
        # Import module for fitting
        rfmodel = RandomForestClassifier(
            n_estimators=80, max_depth=50  # class_weight= {0:.1, 1:.9},
        )

        # Fit the model using the training data
        rfmodel.fit(X_train, y_train)

        return rfmodel

    if solver == 'XGBoost':
        print("training xgboost")
        xgc = xgb.XGBClassifier(n_estimators=500, max_depth=5, base_score=0.5,
                                objective='binary:logistic', random_state=42)
        xgc.fit(X_train, y_train)

        return xgc

    if solver == 'Neural Network':
        # TODO neural network

        from keras.layers import Input, Dense, Dropout
        from keras.models import Model

        n_features = X_train.shape[1]
        inputs = Input(shape=(n_features,))
        print(inputs)
        dense1 = Dense(32, activation='relu')(inputs)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation="relu")(dropout2)
        dropout3 = Dropout(0.2)(dense3)
        outputs = Dense(1, activation='sigmoid')(dropout3)
        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss='binary_crossentropy', optimizer='adam')

        model.fit(X_train.values, y_train.values, epochs=20, verbose=0)

        return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        action="store",
        default=os.getcwd() + '/../data/processed/processed_kiva_data.csv',
        help="dataset path",
    )

    parser.add_argument(
        "--solver",
        action="store",
        default="Random Forest",
        help="Select solver from: (1) 'All' (2) 'Random Forest' (3) 'Embeddings' (4) 'Logistic Regression' ",
    )

    parser.add_argument(
        "--encode",
        action="store",
        default="label",
        help="Select encode type from: (1) 'label' (2) 'ohe' ",
    )

    parser.add_argument(
        "--sample",
        action="store",
        default="undersample",
        help="For imbalanced classes: (1) 'undersample' (2) 'oversample' (3) 'None' ",
    )

    args = parser.parse_args()

    # load data
    df_raw = load_data(args.data)[:1000]

    # preprocess
    df, df_disp = preprocessing(df_raw, args)

    # split data
    y = df['Status']
    X = df.drop(['Status'], axis=1)
    y_disp = df_disp['Status']
    X_disp = df_disp.drop(['Status'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_disp, X_test_disp, y_train_disp, y_test_disp = train_test_split(X_disp, y_disp, test_size=0.2)

    # select model
    option = st.selectbox(
        'Select model',
        ('Random Forest', 'XGBoost', 'Neural Network'),
        index=2 # default to neural network
    )

    st.write('Fit model:', option)

    # model
    model = build_model(option, X_train, y_train)

    # show only one category or not
    category = 0
    if option is 'XGBoost':
        category = None

    # explain
    shap = Shap(option, model, X_train[:5000], category=category)
    shap.explain(X_test[:5000])
    shap.plot_summary(X_test[:5000])
    st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

    # explain first row
    index = 10
    shap.plot_force(X_test_disp.iloc[index, :], index)
    st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

    # print shap values
    shap_values = shap.get_shap_values()
    indexer = shap_values[index]
    st.write(indexer)

    # TODO format data to UI required


if __name__ == '__main__':
    main()
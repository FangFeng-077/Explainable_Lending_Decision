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
    # # visualize processed data
    # st.subheader('Sampled data')
    # st.write(df)

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

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def build_model(X_train, y_train):

    # select model
    model_selected = st.sidebar.selectbox(
        'Select model',
        ('Random Forest', 'XGBoost', 'Neural Network'),
        index=1 # default to neural network
    )
    print(model_selected)

    model = None

    if model_selected == 'Random Forest':
        print("training random forest")
        from model.RandomForest import RandomForest
        model = RandomForest()
        model.fit(X_train, y_train)
        model = model.get_model()

    if model_selected == 'XGBoost':
        print("training xgboost")
        model = xgb.XGBClassifier(n_estimators=500, max_depth=5, base_score=0.5,
                                objective='binary:logistic', random_state=42)
        model.fit(X_train, y_train)

    if model_selected == 'Neural Network':
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

    return model, model_selected

@st.cache(suppress_st_warning=True)
def persist_data(data):
    for k, v in data.items():
        if isinstance(v, str):
            new_v = st.sidebar.selectbox('Select {0}'.format(k), options=[v], index=0)
        else:
            new_v = st.sidebar.slider("Select {0}".format(k), min_value=v * 0.5, max_value=v * 1.5,
                                      value=v * 1.0)
        if new_v:
            data[k] = v
    return data

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

    # streamlit
    st.sidebar.markdown('# model')

    # model
    model, model_selected = build_model(X_train, y_train)

    # index input
    index = st.sidebar.number_input("Select the user index: ", value=10, format="%d")
    if index > 0:
        print('index')

        # user data
        data = X_test_disp.iloc[index, :].copy()
        # st.table(data)

        persist_data(data)

        # explain button
        explain_button = st.sidebar.button('Explanation')

        if explain_button:

            # show only one category or not, 1 means paid
            category = 1
            if model_selected is 'XGBoost':
                category = None

            # explain summary
            shap = Shap(model_selected, model, X_train[:5000], category=category)
            shap.explain(X_test[:5000])
            shap.plot_summary(X_test[:5000])
            st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

            shap.plot_force(data, index)
            st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
            # shap.plot_forces(data, index)
            # st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

            # print shap values
            shap_values = shap.get_shap_values()
            indexer = shap_values[index]
            st.write(indexer)


if __name__ == '__main__':
    main()

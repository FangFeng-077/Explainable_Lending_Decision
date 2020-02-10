import streamlit as st
from model.RandomForest import RandomForest
from model.XGBoost import XGBoost
from model.NeuralNetwork import NeuralNetwork
from util.Preprocessor import Preprocessor
from explain.Shap import Shap
from sklearn.model_selection import train_test_split
import argparse
from performance.performances_func import *
import os

st.title('Explainable Lending Decision')


DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz')


@st.cache
def load_data(path, nrows=None):
    print('load raw data')
    data = pd.read_csv(path, nrows=nrows)
    return data


@st.cache()
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

    # visualize data
    df_disp = df.copy()

    # normalize
    df = preprocessor.normalize(df, normalize_columns)

    # encode category data
    if config.encode == 'ohe':
        # ohe encode
        df = preprocessor.ohe_encode(df, categorical_columns, ordinal_columns)
    elif config.encode == 'label':
        # label encode
        df = preprocessor.label_encode(df, categorical_columns)

    # split data
    y = df['Status']
    X = df.drop(['Status'], axis=1)
    y_disp = df_disp['Status']
    X_disp = df_disp.drop(['Status'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_disp, X_test_disp, y_train_disp, y_test_disp = train_test_split(X_disp, y_disp, test_size=0.2)

    return X_train, X_test, y_train, y_test, X_train_disp, X_test_disp, y_train_disp, y_test_disp


def build_model(X_train, y_train):
    print('build model')

    # select model
    model_selected = st.sidebar.selectbox(
        'Select model',
        ('Random Forest', 'XGBoost', 'Neural Network'),
        index=1 # default to neural network
    )

    model = None

    if model_selected == 'Random Forest':
        print("training random forest")
        model = RandomForest()

    if model_selected == 'XGBoost':
        print("training xgboost")
        model = XGBoost()

    if model_selected == 'Neural Network':
        print("training Neural Network")
        model = NeuralNetwork()

    model.fit(X_train, y_train)

    return model.get_model(), model.get_title()


@st.cache()
def load_user_data(index, data):
    print('load user data')
    return data.iloc[index, :].copy()


def update_feature(data):
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
        "--encode",
        action="store",
        default="label",
        help="Select encode type from: (1) 'label' (2) 'ohe' ",
    )

    parser.add_argument(
        "--sample",
        action="store",
        default="undersample",
        help="Select sample type from: (1) 'undersample' (2) 'oversample' ",
    )

    args = parser.parse_args()

    # load data
    df_raw = load_data(args.data)[:1000]

    # preprocess
    X_train, X_test, y_train, y_test, X_train_disp, X_test_disp, y_train_disp, y_test_disp = preprocessing(df_raw, args)

    # streamlit
    st.sidebar.markdown('# Model')

    # model
    model, title = build_model(X_train, y_train)

    # prediction
    y_pred = predict_labels(model, X_test, y_test)

    # get performace
    performance = get_metrics(y_test, y_pred)

    st.subheader('model performance')
    st.table(performance)

    # index input
    # index = st.sidebar.number_input("Select the user index: ", value=10, format="%d")

    # if index > 0:
    index = 10

    # user data
    data = load_user_data(index, X_test_disp)
    # st.table(data)

    update_feature(data)

    # explain button
    explain_button = st.sidebar.button('Explanation')

    if explain_button:

        background_data = X_train[:5000].copy()
        explain_data = X_test[:500].copy()

        if title == 'Neural Network':
            background_data = background_data.reset_index(drop=True).to_numpy()
            explain_data = explain_data.reset_index(drop=True).to_numpy()

        # explain summary
        shap = Shap(title, model, background_data)
        shap.explain(explain_data)
        shap.plot_summary(explain_data)
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

        shap.plot_force(data, index)
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

        # shap.plot_image(data, index)
        # st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

        # print shap values
        shap_values = shap.get_shap_values()
        indexer = shap_values[index]
        st.write(indexer)


if __name__ == '__main__':
    main()

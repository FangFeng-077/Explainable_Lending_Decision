import streamlit as st
import altair as alt
from src.model.RandomForest import RandomForest
from src.model.XGBoost import XGBoost
from src.model.NeuralNetwork import NeuralNetwork
from src.util.Preprocessor import Preprocessor
from src.explain.Shap import Shap
from sklearn.model_selection import train_test_split
import argparse
from src.util.performances_func import *
from src.util.Config import Config
import os

st.title('Explainable Lending Decision')

@st.cache
def load_data(path, nrows=None):
    data = pd.read_csv(path, nrows=nrows)
    return data


@st.cache()
def preprocessing(df, config):
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
    X_train, X_test, \
    y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_disp = df_disp['Status']
    X_disp = df_disp.drop(['Status'], axis=1)
    X_train_disp, X_test_disp, \
    y_train_disp, y_test_disp = train_test_split(X_disp, y_disp, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X_train_disp, X_test_disp, y_train_disp, y_test_disp


def build_model(model_selected, X_train, y_train):

    model = None

    if model_selected == 'Random Forest':
        model = RandomForest()

    if model_selected == 'XGBoost':
        model = XGBoost()

    if model_selected == 'Neural Network':
        model = NeuralNetwork(X_train.shape[1])

    if not model:
        return None

    model.fit(X_train, y_train)

    return model, model.get_title()


def update_feature(data):
    for k, v in data.items():
        k_repr = k.replace(" ", "")
        if not Config.has(k_repr):
            continue

        if isinstance(v, str):
            new_v = st.sidebar.selectbox('Select {0}'.format(k),
                                         options=Config.get(k_repr),
                                         index=0)
        else:
            if k == 'Partner ID':
                new_v = st.sidebar.slider("Select {0}".format(k),
                                          min_value=int(v * 0.5),
                                          max_value=int(v * 1.5),
                                          value=int(v))
            else:
                new_v = st.sidebar.slider("Select {0}".format(k),
                                          min_value=v * 0.5,
                                          max_value=v * 1.5,
                                          value=v * 1.0)
        if new_v:
            data[k] = v
    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        action="store",
        default=os.getcwd() + '/data/raw/kiva_data.csv',
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
    X_train, X_test, y_train, y_test, \
    X_train_disp, X_test_disp, y_train_disp, y_test_disp = preprocessing(df_raw, args)

    # streamlit
    st.sidebar.markdown('# Model')

    # select model
    model_selected = st.sidebar.selectbox(
        'Select model',
        ('Random Forest', 'XGBoost', 'Neural Network'),
        index=0 # default to neural network
    )

    # model
    model, title = build_model(model_selected, X_train, y_train)

    # prediction
    y_pred = predict_labels(model, X_test, y_test)

    # get performace
    # performance = get_metrics(y_test, y_pred)

    # st.subheader('model performance')
    # st.table(performance)

    # index input
    max_index = X_test_disp.shape[0] - 1
    index = st.sidebar.number_input("Select the user index from 0 to {}: ".format(max_index),
                                    value=10,
                                    min_value=0,
                                    max_value=max_index,
                                    format="%d")

    # user data
    data = X_test_disp.iloc[index, :].copy()

    # st.table(data)

    data = update_feature(data)

    # explain button
    explain_button = st.sidebar.button('Explanation')

    if explain_button:

        indexes = list(data.index)
        user_pred = y_pred.iloc[index]

        if user_pred == 1:
            st.markdown('Prediction: Paid :smile:')
        else:
            st.markdown('Prediction: Default :confused:')

        st.write("")

        background_data = X_train[:5000].copy()
        explain_data = X_test[:500].copy()

        if title == 'Neural Network':
            background_data = background_data.reset_index(drop=True).to_numpy()
            explain_data = explain_data.reset_index(drop=True).to_numpy()

        # explain summary
        shap = Shap(title, model.get_model(), background_data)
        shap.explain(explain_data)
        shap_values = shap.get_shap_values()

        # visualization
        user_shap_values = shap_values[index]
        pivot = np.sum(np.abs(user_shap_values))
        user_shap_prob = np.round(user_shap_values / pivot, 2)
        user_shap_prob_df = pd.DataFrame({'index': indexes, 'probability': user_shap_prob})

        bars = alt.Chart(user_shap_prob_df).mark_bar().encode(
            x='index',
            y=alt.X('probability', axis=alt.Axis(format='%', title='Influence Factor')),
            color=alt.condition(
                alt.datum.probability >= 0,  # If the year is 1810 this test returns True,
                alt.value('#F63366'),  # which sets the bar orange.
                alt.value('#1E88E5')  # And if it's not true it sets the bar steelblue.
            )
        ).properties(width=640, height=480)

        text_above = bars.transform_filter(alt.datum.probability >= 0).mark_text(
            align='center',
            baseline='bottom',
            dy=-3
        ).encode(text='probability')

        text_below = bars.transform_filter(alt.datum.probability < 0).mark_text(
            align='center',
            baseline='bottom',
            dy=15
        ).encode(text='probability')

        st.altair_chart(bars + text_above + text_below)

        # summary plot
        st.subheader("Global Interpretation")
        shap.plot_summary(explain_data)
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

        # force plot
        st.subheader("Local Interpretation")
        shap.plot_force(data, index)
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)



if __name__ == '__main__':
    main()
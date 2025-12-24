"""
Iris app data project

Class: Advanced Python for Data Science
The aim of this app is to load the iris dataset, visualize it and predict the flower species
based on user input using a Random Forest model.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


@st.cache_data
def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Loads the Iris dataset from sklearn and converts it into a pandas DataFrame.

    The numerical target values are mapped to their corresponding species names.

    :return: A tuple containing:
        - A pandas DataFrame with iris features and species column
        - A list of species names
    :rtype: tuple
    """
    iris_data_bunch = load_iris()  # pylint: disable=no-member

    data_frame = pd.DataFrame(
        iris_data_bunch.data,  # pylint: disable=no-member
        columns=iris_data_bunch.feature_names  # pylint: disable=no-member
    )

    data_frame["species"] = pd.Series(iris_data_bunch.target).map(  # pylint: disable=no-member
        dict(enumerate(iris_data_bunch.target_names))  # pylint: disable=no-member
    )

    return data_frame, iris_data_bunch.target_names  # pylint: disable=no-member


@st.cache_resource
def train_model(x_data: pd.DataFrame, y_target: pd.Series) -> RandomForestClassifier:
    """
    Trains a random forest classifier on the provided feature data (X) and target (y).
    :param x_data: The measurements.
    :param y_target: The species.
    :return: The trained model.
    """
    model = RandomForestClassifier()
    model.fit(x_data, y_target)
    return model


def user_input_features(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the sidebar interface for user input parameters
    and returns the input as a single-row DataFrame.
    :param data_frame: The full Iris DataFrame used to set slider min/max bounds.
    :return: DataFrame containing the user's input features.
    :rtype: pd.DataFrame
    """
    st.sidebar.header("User Input Parameters for Prediction")

    sl_min, sl_max = data_frame["sepal length (cm)"].min(), data_frame["sepal length (cm)"].max()
    sw_min, sw_max = data_frame["sepal width (cm)"].min(), data_frame["sepal width (cm)"].max()
    pl_min, pl_max = data_frame["petal length (cm)"].min(), data_frame["petal length (cm)"].max()
    pw_min, pw_max = data_frame["petal width (cm)"].min(), data_frame["petal width (cm)"].max()

    data = {
        "sepal length (cm)": st.sidebar.slider("Sepal Length (cm)", sl_min, sl_max, 5.4),
        "sepal width (cm)": st.sidebar.slider("Sepal Width (cm)", sw_min, sw_max, 3.4),
        "petal length (cm)": st.sidebar.slider("Petal Length (cm)", pl_min, pl_max, 1.3),
        "petal width (cm)": st.sidebar.slider("Petal Width (cm)", pw_min, pw_max, 0.2),
    }
    return pd.DataFrame(data, index=[0])


def run_app() -> None:
    """
    Main function to run the Streamlit application
    """
    st.set_page_config(page_title="Iris App", layout="wide")
    iris_df, species_names = load_data()

    st.header("🔬 Iris Dataset Explorer")
    st.write(iris_df.head())

    st.subheader("Interactive Scatter Plot")
    col_names = iris_df.columns[:-1].tolist()
    x_axis = st.selectbox("Select X-axis feature", col_names, index=2)
    y_axis = st.selectbox("Select Y-axis feature", col_names, index=3)

    fig = px.scatter(
        iris_df,
        x=x_axis,
        y=y_axis,
        color="species",
        title=f"{x_axis.title()} vs. {y_axis.title()} by Species",
    )
    st.plotly_chart(fig, use_container_width=True)

    input_df = user_input_features(iris_df)
    st.subheader("User Input:")
    st.write(input_df)

    x_train = iris_df.iloc[:, :-1]
    y_train = iris_df["species"]
    clf = train_model(x_train, y_train)

    prediction_raw = clf.predict(input_df)
    prediction_proba = clf.predict_proba(input_df)

    st.subheader("Prediction:")
    st.success(f"The predicted Iris Species is: **{prediction_raw[0].title()}**")

    st.subheader("Prediction Probability by Species:")
    proba_df = pd.DataFrame(prediction_proba, columns=species_names)
    st.dataframe(proba_df.T.rename(columns={0: "Probability"}), use_container_width=True)


if __name__ == "__main__":
    run_app()

import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier, mo


@app.cell
def _(mo):
    mo.md("""
    # Fitting and predicting: estimator basic
    """)
    return


@app.cell
def _(RandomForestClassifier):
    clf = RandomForestClassifier(random_state=42)

    # 2 samples, 3 features
    X = [[1, 2, 3], [11, 12, 13]]
    # classes of each sample
    y = [0, 1]
    return X, clf, y


@app.cell
def _(X, clf, y):
    clf.fit(X, y)
    return


@app.cell
def _(X, clf):
    # predict classes of the training data
    clf.predict(X)
    return


@app.cell
def _(clf):
    # predict classes of new data
    clf.predict([[4, 5, 6], [14, 15, 16]])
    return


@app.cell
def _(clf):
    import os
    from pathlib import Path
    from pickle import dump, load
    from sklearn.base import BaseEstimator

    MODEL_BASE_DIR = 'models/getting_started/'

    # save the model
    def save_pickle_model(name: str, model: BaseEstimator, base_dir: str = MODEL_BASE_DIR):
        """Save ML model as pickle file"""
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        cwd = Path(os.getcwd())
        with open(cwd / Path(base_dir + name), "wb") as f:
            dump(model, f, protocol=5)
        return True

    save_pickle_model('intro_model.pkl', clf)

    # load and use the model
    def load_pickle_model(name: str, base_dir: str = MODEL_BASE_DIR):
        cwd = Path(os.getcwd())
        with open(cwd / Path(MODEL_BASE_DIR + name), "rb") as file:
            clf_load = load(file)
        return clf_load

    return load_pickle_model, save_pickle_model


@app.cell
def _(load_pickle_model):
    # use saved model
    clf_load = load_pickle_model('intro_model.pkl')
    clf_load.predict([[4, 5, 6], [14, 15, 16]])
    return


@app.cell
def _(mo):
    mo.md("""
    # Transformers and pre-processors
    """)
    return


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler

    return (StandardScaler,)


@app.cell
def _(StandardScaler):
    X1 = [[0, 15], [1,-10]]
    # scale data according to computed scaling values
    scaled_values = StandardScaler().fit(X1).transform(X1)
    return


@app.cell
def _(mo):
    mo.md("""
    # Column transformer for heterogeneous data
    """)
    return


@app.cell
def _():
    import pandas as pd

    X3 = pd.DataFrame(
        {
            "city": ["London", "London", "Paris", "Sallisaw"],
            "title": [
                "His Last Bow",
                "How Watson Learned the Trick",
                "A Moveable Feast",
                "The Grapes of Wrath",
            ],
            "expert_rating": [5, 3, 4, 5],
            "user_rating": [4, 5, 4, 3],
        }
    )
    return (X3,)


@app.cell
def _(X3):
    X3.info()
    return


@app.cell
def _():
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer, CountVectorizer, OneHotEncoder


@app.cell
def _(ColumnTransformer, CountVectorizer, OneHotEncoder):
    column_trans = ColumnTransformer(
        [
            ("categories", OneHotEncoder(dtype="int"), ["city"]),
            ("title_bow", CountVectorizer(), "title"),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return (column_trans,)


@app.cell
def _(X3, column_trans):
    column_trans.fit(X3)
    return


@app.cell
def _(column_trans):
    column_trans.get_feature_names_out()
    return


@app.cell
def _(X3, column_trans):
    column_trans.transform(X3).toarray()
    return


@app.cell
def _(mo):
    mo.md("""
    # Pipelines: chaining pre-processors and estimators
    """)
    return


@app.cell
def _():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    return (
        LogisticRegression,
        accuracy_score,
        load_iris,
        make_pipeline,
        train_test_split,
    )


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    load_iris,
    make_pipeline,
    train_test_split,
):
    # create a pipeline object
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )

    # load the iris dataset and split it into train and test sets (70:30)
    X_data, y_data = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.3, random_state=0)

    # fit the whole pipeline
    pipe.fit(X_train, y_train)
    return X_test, X_train, pipe, y_test, y_train


@app.cell
def _(X_test, accuracy_score, pipe, y_test):
    accuracy_score(pipe.predict(X_test), y_test)
    return


@app.cell
def _(pipe, save_pickle_model):
    # save the model
    save_pickle_model('pipeline_model.pkl', pipe)
    return


@app.cell
def _(mo):
    mo.md("""
    # Model evaluation
    """)
    return


@app.cell
def _():
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_validate, cross_val_score

    return LinearRegression, cross_val_score, cross_validate, make_regression


@app.cell
def _(LinearRegression, make_regression):
    X4, y4 = make_regression(n_samples=1000, random_state=0)
    lr = LinearRegression()
    return X4, lr, y4


@app.cell
def _(X4, cross_validate, lr, y4):
    result = cross_validate(lr, X4, y4, cv=10) # defaults to 5-fold CV
    result['test_score'] # r_squared score is high because dataset is easy
    return


@app.cell
def _(X_train, cross_val_score, pipe, y_train):
    # Evaluate against iris dataset
    scores = cross_val_score(pipe, X_train, y_train, cv=10)
    print(scores)
    return


@app.cell
def _(mo):
    mo.md("""
    # Automatic parameter searches
    """)
    return


@app.cell
def _():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    return RandomForestRegressor, RandomizedSearchCV, randint


@app.cell
def _(make_regression):
    # create a synthetic dataset
    X5, y5 = make_regression(
        n_samples=20640, n_features=8, noise=0.1, random_state=0
    )
    return X5, y5


@app.cell
def _(X5, train_test_split, y5):
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X5, y5, random_state=0)
    return X_test2, X_train2, y_test2, y_train2


@app.cell
def _(randint):
    # define the parameter space that will be searched over
    param_distributions = {
        "n_estimators": randint(1, 5),
        "max_depth": randint(5, 10),
    }
    return (param_distributions,)


@app.cell
def _(
    RandomForestRegressor,
    RandomizedSearchCV,
    X_train2,
    param_distributions,
    y_train2,
):
    # now create a searchCV object and fit it to the data
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=0),
        n_iter=5,
        param_distributions=param_distributions,
        random_state=0,
    )

    search.fit(X_train2, y_train2)
    return (search,)


@app.cell
def _(search):
    search.best_params_
    return


@app.cell
def _(X_test2, search, y_test2):
    # the search object now acts like a normal random forest estimator
    # with max_depth=9 and n_estimators=4
    search.score(X_test2, y_test2)
    return


@app.cell
def _(save_pickle_model, search):
    # save model
    save_pickle_model('randomized_search_model.pkl', search)
    return


if __name__ == "__main__":
    app.run()

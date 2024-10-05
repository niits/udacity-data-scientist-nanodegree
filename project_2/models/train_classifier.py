import pickle
import re
import sys

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt_tab")


def load_data(database_filepath: str) -> tuple[pd.Series, pd.DataFrame, pd.Index]:
    """
    Load data from the SQLite database.

    Args:
        database_filepath (str): Path to the SQLite database file.

    Returns:
        tuple: A tuple containing:
            - X (pd.Series): Series containing the messages.
            - Y (pd.DataFrame): DataFrame containing the categories for each message.
            - category_names (Index): Index object containing the category names.
    """
    engine = create_engine("sqlite:///" + database_filepath)

    df = pd.read_sql_table("messages", engine)
    X = df["message"]
    Y = df.iloc[:, 4:]

    category_names = Y.columns

    return X, Y, category_names


def tokenize(text: str) -> str:
    """
    Tokenizes and cleans a given text string.

    This function performs the following steps:
    1. Replaces URLs in the text with a placeholder string "urlplaceholder".
    2. Converts the text to lowercase and removes any non-alphanumeric characters.
    3. Tokenizes the cleaned text into individual words.
    4. Lemmatizes each token to its base form.
    5. Strips any leading or trailing whitespace from each token.

    Args:
        text (str): The text string to be tokenized and cleaned.

    Returns:
        List[str]: A list of cleaned and lemmatized tokens.
    """
    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"  # noqa
    text = re.sub(url_regex, "urlplaceholder", text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]

    return clean_tokens


def build_model():
    """
    Build a machine learning model pipeline and perform hyperparameter tuning using GridSearchCV.

    The pipeline includes the following steps:
    - CountVectorizer: Converts a collection of text documents to a matrix of token counts.
    - TfidfTransformer: Transforms the count matrix to a normalized tf-idf representation.
    - MultiOutputClassifier: Fits a separate classifier for each output variable.

    The hyperparameters for tuning include:
    - vect__max_df: Maximum document frequency for the CountVectorizer.
    - vect__ngram_range: The range of n-values for different n-grams to be extracted.
    - tfidf__use_idf: Whether to use inverse document frequency reweighting.
    - clf__estimator__n_estimators: Number of trees in the RandomForestClassifier.
    - clf__estimator__min_samples_split: Minimum number of samples required to split an internal node.
    - clf__estimator__max_depth: Maximum depth of the tree.
    - clf__estimator__min_samples_leaf: Minimum number of samples required to be at a leaf node.
    - clf__estimator__bootstrap: Whether bootstrap samples are used when building trees.

    Returns:
        GridSearchCV: A GridSearchCV object configured with the pipeline and parameter grid.
    """

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize, token_pattern=None)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    parameters = {
        "vect__max_df": [0.75, 1.0],
        "clf__estimator__n_estimators": [50, 100],
        "clf__estimator__min_samples_split": [2, 4],
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2, cv=3)
    return model


def evaluate_model(
    model: GridSearchCV,
    X_test: pd.Series,
    Y_test: pd.DataFrame,
    category_names: pd.Index,
) -> None:
    """
    Evaluate the model's performance on the test data and print the classification report.

    Args:
        model (GridSearchCV): The trained model to be evaluated.
        X_test (pd.Series): The test data features.
        Y_test (pd.DataFrame): The true labels for the test data.
        category_names (pd.Index): The names of the categories.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """
    Save the trained model as a pickle file.

    Args:
        model (GridSearchCV): The trained model to be saved.
        model_filepath (str): The file path where the model will be saved.

    Returns:
        None
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()

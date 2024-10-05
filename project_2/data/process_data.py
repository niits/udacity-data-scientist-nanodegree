import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load messages and categories datasets and merge them on 'id'.

    Args:
        messages_filepath (str): File path to the messages CSV file.
        categories_filepath (str): File path to the categories CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing messages and categories.
    """
    messages_df = pd.read_csv(messages_filepath, index_col="id")
    categories_df = pd.read_csv(categories_filepath, index_col="id")
    return messages_df.join(categories_df)


def extract_key_value(string_value: str) -> dict:
    """
    Extract key-value pairs from a string in the format 'key-value'.

    Args:
        string_value (str): String containing key-value pairs separated by semicolons.

    Returns:
        dict: Dictionary with keys and integer values extracted from the string.
    """
    data = {}
    for substring in string_value.split(";"):
        parts = substring.split("-")
        if len(parts) == 2:
            data[parts[0]] = int(parts[1])
        else:
            raise ValueError(f"{substring} is not in defined form")
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged DataFrame by splitting categories into separate columns.

    Args:
        df (pd.DataFrame): Merged DataFrame containing messages and categories.

    Returns:
        pd.DataFrame: Cleaned DataFrame with categories split into separate columns.
    """
    categories = df["categories"].apply(extract_key_value).apply(pd.Series)
    return pd.concat([df.drop("categories", axis=1), categories], axis=1)


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
    Save the cleaned DataFrame to an SQLite database.

    Args:
        df (pd.DataFrame): Cleaned DataFrame to be saved.
        database_filename (str): File path to the SQLite database file.

    Returns:
        None
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("messages", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()

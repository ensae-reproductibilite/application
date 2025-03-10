def split_and_count(df, column, separator):
    """
    Split a column in a DataFrame by a separator and count the number of resulting elements.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to split.
        column (str): The name of the column to split.
        separator (str): The separator to use for splitting.

    Returns:
        pandas.Series: A Series containing the count of elements after splitting.

    """
    return df[column].str.split(separator).str.len()

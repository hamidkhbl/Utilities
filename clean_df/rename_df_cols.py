def rename_df_cols(data_frame):
    """
        input: a pandas dataframe
    output: a pandas dataframe with better column names

    This method replaces space with '_' for columns of a dataframe
    """
    new_cols = []
    for col in data_frame.columns:
        new_cols.append(col.strip().replace(' ', '_'))
    data_frame.columns = new_cols

    return data_frame

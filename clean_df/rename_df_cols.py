import pandas

def rename_df_cols(df):
    '''
    input: a pandas dataframe
    output: a pandas dataframe with better column names

    This method replaces space with '_' for columns of a dataframe
    '''
    new_cols = []
    for col in df.columns:
        new_cols.append(col.strip().replace(' ','_'))
    df.columns = new_cols

    return df

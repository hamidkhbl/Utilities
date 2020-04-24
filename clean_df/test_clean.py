from rename_df_cols import rename_df_cols
import pandas as pd

def test_rename_cols_space():
    df = pd.DataFrame(data = {'test col':'test value'}, columns=['test test','something'])
    df2 = pd.DataFrame(data = {'test col':'test value'}, columns=['test_test','something'])
    assert(rename_df_cols(df).equals(df2))

def test_rename_cols_space_strip():
    df = pd.DataFrame(data = {'test col':'test value'}, columns=[' test test',' something'])
    df2 = pd.DataFrame(data = {'test col':'test value'}, columns=['test_test','something'])
    assert(rename_df_cols(df).equals(df2))
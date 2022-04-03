import numpy as np
import pandas as pd

import pygsheets

def write_to_gsheet(service_file_path, spreadsheet_id, sheet_name, data_df):
    """
    this function takes data_df and writes it under spreadsheet_id
    and sheet_name using your credentials under service_file_path
    """
    gc = pygsheets.authorize(service_file=service_file_path)
    sh = gc.open_by_key(spreadsheet_id)
    try:
        sh.add_worksheet(sheet_name)
    except:
        pass
    wks_write = sh.worksheet_by_title(sheet_name)
    wks_write.clear('A1',None,'*')
    wks_write.set_dataframe(data_df, (1,1), encoding='utf-8', fit=True)
    wks_write.frozen_rows = 1



def main():
    df = pd.read_csv("Data.csv")
    write_to_gsheet("jsonFileFromGoogle.json", "1exhQ6oXQ38yLZNZG380VErZnp20vWTI8i4tzEbqw8pE", "Sheet1", df)


main()
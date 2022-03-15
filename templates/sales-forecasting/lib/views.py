import pandas as pd
import numpy as np


def make_pivot_table(df, value):
    dg  = pd.pivot_table(df, values=value, index=['Order Date'],
                    columns=['Category', 'Sub-Category', 'Product Name', 'City'])
    dg = dg.resample('M').sum() 
    dg = dg.sort_index(ascending=False)
    dg.index = dg.index.strftime('%B  %Y')
    
    return dg.transpose()




    
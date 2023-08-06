import itertools
import pandas as pd

def load_data(fname):
    df = pd.read_csv('./ftl.csv')
    f, t, l = (df['F'].dropna().drop_duplicates(),
               df['T'].dropna().drop_duplicates(),
               df['L'].dropna().drop_duplicates())
    # return the cartesian product of the columns
    ftl = pd.DataFrame(list(itertools.product(f, t, l)), columns=['F', 'T', 'L'])
    ftl['FTL'] = ftl[['F','T','L']].agg(','.join, axis=1)
    return ftl

if __name__ == '__main__':
    num_print = 4000
    ftl = load_data('./ftl.csv').sample(frac=1)
    
    for i in range(num_print):
        print(ftl['FTL'].iloc[i])


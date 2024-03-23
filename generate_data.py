import argparse
import itertools
import pandas as pd

def load_data(fname):
    df = pd.read_csv(fname)
    f, t, l = (df['f'].dropna().drop_duplicates(),
               df['t'].dropna().drop_duplicates(),
               df['l'].dropna().drop_duplicates())
    return f, t, l

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_all', action='store_true')
    args = parser.parse_args()

    if args.generate_all:
        f, t, l = load_data('./ftl.csv').sample(frac=1)
        # cartesian product of the columns is every possible acronym
        ftl = pd.DataFrame(list(itertools.product(f, t, l)), columns=['f', 't', 'l'])
        ftl.to_csv('./ftl-all.csv', index=False)


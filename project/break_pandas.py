import pandas as pd
import glob
import time

start = time.time()
files = glob.glob("data/*.gz")
df = pd.concat([pd.read_csv(f, sep=' ', names=['domain','title','views','size'], compression='gzip') for f in files])
print(f"Loaded {len(df)} rows in {time.time() - start:.2f} seconds")

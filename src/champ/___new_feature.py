import os

fs = os.listdir()

for f in fs:
    if " " in f:
        print(f)
        os.rename(f, f.replace(" ", "_"))
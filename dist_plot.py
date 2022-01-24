import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

df = pd.read_csv(Path("run_dist/res_dist.csv"), sep=",", names=["algo", "score"])

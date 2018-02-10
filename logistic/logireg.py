import numpy as np
import pandas as pd

def show_data():
  data = pd.read_csv('LogiReg_data.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
  
  
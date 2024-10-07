import os
import sys
import numpy as np
import torch
import random
import pandas as pd
import importlib
import pickle
from IPython.display import Image
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def setup_environment(notebook_globals):
    current_dir = os.getcwd()  # Get the current working directory
    parent_dir = os.path.join(os.path.dirname(current_dir), 'src') 
    print(parent_dir)

    # Add paths to source code and load it
    paths = [
        os.path.join(parent_dir, "data"),
        os.path.join(parent_dir, "utils"),
        os.path.join(parent_dir, "training"),
        os.path.join(parent_dir, "mld"),
        os.path.join(parent_dir, "mld/libdesign"),
        os.path.join(parent_dir, "models"),
        
    ]

    for path in paths:
        if path not in sys.path:
            sys.path.append(path)

    # Print paths to verify they are correct
    print("Paths added to sys.path:")
    for path in paths:
        print(path)

    # Import modules
    notebook_globals['np'] = np
    notebook_globals['torch'] = torch
    notebook_globals['random'] = random
    notebook_globals['pd'] = pd
    notebook_globals['Image'] = Image
    notebook_globals['stats'] = stats
    notebook_globals['sns'] = sns
    notebook_globals['plt'] = plt

    import sequence_utils
    import seq2fitness_train
    import seq2fitness_traintools
    import seq2fitness_models

    # Add imported modules to notebook_globals
    notebook_globals['sequence_utils'] = sequence_utils
    notebook_globals['train'] = seq2fitness_train
    notebook_globals['traintools'] = seq2fitness_traintools
    notebook_globals['seq2fitness_models'] = seq2fitness_models

    # Define additional strings or variables if needed
    ref_seq_amylase = """LTAPSIKSGTILHAWNWSFNTLKHNMKDIHDAGYTAIQTSPINQVKEGNQGDKSMSNWYWLYQPTSYQIGNRYLGTEQEFKEMCAAAEEYGIKVIVDAVINHTTSDYAAIS
NEVKSIPNWTHGNTPIKNWSDRWDVTQNSLSGLYDWNTQNTQVQSYLKRFLDRALNDGADGFRFDAAKHIELPDDGSYGSQFWPNITNTSAEFQYGEILQDSVSRDAAYANY
MDVTASNYGHSIRSALKNRNLGVSNISHYAVDVSADKLVTWVESHDTYANDDEESTWMSDDDIRLGWAVIASRSGSTPLFFSRPEGGGNGVRFPGKSQIGDRGSALFEDQAI
TAVNRFHNVMAGQPEELSNPNGNNQIFMNQRGSHGVVLANAGSSSVSINTATKLPDGRYDNKAGAGSFQVNDGKLTGTINARSVAVLYPD""".replace('\n','')
    notebook_globals['ref_seq_amylase'] = ref_seq_amylase

# Call the setup function if this script is executed
if __name__ == "__main__":
    setup_environment(globals())

from util import get_data
import numpy as np

split = get_data()
language = [comparison['language'] for comparison in split]
np.save("language_users.npy", np.array(language))

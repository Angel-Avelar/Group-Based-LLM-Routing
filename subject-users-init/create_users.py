from util import get_data
import numpy as np
import pandas as pd
from config import bool_cols

split = get_data()

rows = [
    {
        'creative_writing': comparison['category_tag']['creative_writing_v0.1']['creative_writing'],
        **{k: comparison['category_tag']['criteria_v0.1'][k] for k in 
           ['complexity', 'creativity', 'domain_knowledge', 'problem_solving', 
            'real_world', 'specificity', 'technical_accuracy']},
        'math': comparison['category_tag']['math_v0.1']['math'],
        'instruction_following': comparison['category_tag']['if_v0.1']['if'],
        'is_code': comparison['is_code'],
        'language': comparison['language']
    }
    for comparison in split
]

df = pd.DataFrame(rows)

df['user'] = df[bool_cols].apply(
    lambda row: int(''.join(row.astype(int).astype(str)), 2),
    axis=1
)

np.save("category_users.npy", np.array(df['user']))

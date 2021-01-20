from pathlib import Path

import pandas as pd

from digit_recognition.config import raw_train_path, split_dir, \
    split_test_y_true, split_test, split_train
from digit_recognition.custom_funcs import initial_sss


if split_test_y_true.exists() and split_test.exists() and split_train.exists():
    print(f'Split data is already present in: \n {split_dir}')
else:
    print('Proceeding ot split data: ')
    df = pd.read_csv(raw_train_path)
    initial_sss(df, 'label', 0.2, split_dir)

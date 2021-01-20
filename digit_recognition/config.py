from pathlib import Path

current_wd = Path(__file__).resolve()
current_wd = current_wd.parent.parent

sample_sub_path = current_wd / 'data/raw/sample_submission.csv'
raw_test_path = current_wd / 'data/raw/test.csv'
raw_train_path = current_wd / 'data/raw/train.csv'

split_dir = current_wd / 'data/split'
split_train = current_wd / 'data/split/train_df.csv'
split_test = current_wd / 'data/split/test_df.csv'
split_test_y_true = current_wd / 'data/split/test_df_y_true.csv'

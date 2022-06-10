# create function to split the dataset into train and test
import random

from sklearn.model_selection import GroupShuffleSplit


def split_dataset(dataframe, test_size):
    splitter = GroupShuffleSplit(test_size=test_size, random_state=42)
    split = splitter.split(dataframe, groups=dataframe['image_path'])
    train_inds, test_inds = next(split)

    return dataframe.iloc[train_inds], dataframe.iloc[test_inds]


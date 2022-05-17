import numpy as np   
import pandas as pd
from scipy.stats import skew
from scipy.signal import find_peaks

from sklearn.base import BaseEstimator, TransformerMixin


class DFToDictTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.to_dict('records')




def extract_features(X, bases=None):
    """
    Extract features from data.
    :param X: Dataset of time windows.
    :param bases: Dictionary with values of bool lists of size 7 and keys of the names of the vectors to extract
    features from
    :return: new dataset with extracted features, training feature name list
    """
    # Restructure dataframe to fit preprocessing features extraction functions - changing the dataframe to have one row.
    # Each column is compressed to a list.
    data = pd.DataFrame(columns=X.columns)
    for col in data.columns:
        data.loc[0,col]= np.array(X[col])

    if bases is None:
        bases = {
        'RSSI': [True, False, False, False, True, True, True],
        'RSSI_diffs': [True, True, True, False, True, True, True],
        'RSSI_diffs_abs': [False, False, True, True, True, False, True],
        'RSSI_median_dist': [True, False, True, True, True, False, True]
    } 

    features_name = []
    data['RSSI_diffs'] = data.RSSI.apply(lambda x: x[1:] - x[:-1])
    data['RSSI_diffs_abs'] = data.RSSI.apply(lambda x: abs(x[1:] - x[:-1]))
    data['RSSI_median_dist'] = data.RSSI.apply(lambda x: abs(x - np.median(x)))

    data, features_name = extract_list_feats('RSSI', data, features_name, base=bases['RSSI'])
    data, features_name = extract_list_feats('RSSI_diffs', data, features_name, base=bases['RSSI_diffs'])
    data, features_name = extract_list_feats('RSSI_diffs_abs', data, features_name, base=bases['RSSI_diffs_abs'])
    data, features_name = extract_list_feats('RSSI_median_dist', data, features_name, base=bases['RSSI_median_dist'])

    data['max_count_same_value_RSSI'] = data.RSSI.apply(lambda x: np.max(np.unique(x, return_counts=True)[1]))
    features_name += ['max_count_same_value_RSSI']

    data['RSSI_peaks'] = data.RSSI.apply(lambda x: len(find_peaks(x)[0]))
    features_name += ['RSSI_peaks']

    data['RSSI_diffs_peaks'] = data.RSSI_diffs.apply(lambda x: len(find_peaks(x)[0]))
    features_name += ['RSSI_diffs_peaks']

    data['peak_ratio_diffs_RSSI'] = data.apply(
        lambda x: x['RSSI_diffs_peaks'] / x['RSSI_peaks'] if x['RSSI_peaks'] > 0 else 0, axis=1)
    features_name += ['peak_ratio_diffs_RSSI']

    data['RSSI_values_count'] = data.RSSI.apply(lambda x: len(np.unique(x)))
    features_name += ['RSSI_values_count']

    return data, features_name

def extract_list_feats(list_name: str, data, features_name: list, base=None):
    """
    Extract Features from vector.
    :param list_name: Vector to extract features from.
    :param data: Dataset to extract features from.
    :param features_name: Feature list to add new feature names to.
    :param base: Disable the use of features.
    :return: Data with features, updated feature name list.
    """

    if base is None:
        base = DEFAULT_TRUE_LIST

    data[f'max_{list_name}'] = data[list_name].apply(np.max)
    if base[0]:
        features_name += [f'max_{list_name}']

    data[f'min_{list_name}'] = data[list_name].apply(np.min)
    if base[1]:
        features_name += [f'min_{list_name}']

    data[f'mean_{list_name}'] = data[list_name].apply(np.mean)
    if base[2]:
        features_name += [f'mean_{list_name}']

    data[f'median_{list_name}'] = data[list_name].apply(np.median)
    if base[3]:
        features_name += [f'median_{list_name}']

    data[f'std_{list_name}'] = data[list_name].apply(np.std)
    if base[4]:
        features_name += [f'std_{list_name}']

    data[f'skew_{list_name}'] = data[list_name].apply(skew)
    if base[5]:
        features_name += [f'skew_{list_name}']

    data[f'max_sub_min_{list_name}'] = data[list_name].apply(lambda x: np.max(x) - np.min(x))
    if base[6]:
        features_name += [f'max_sub_min_{list_name}']

    return data, features_name
    
def preprocess(X, RSSI_value_selection):
    """
    Calculate the features on the selected RSSI on the test set
    :param X: Dataset to extract features from.
    :param RSSI_value_selection: Which signal values to use- - in our case it is Average.
    :return: Test x dataset with features
    """
    if RSSI_value_selection=="RSSI_Left":
        X["RSSI"] = X.RSSI_Left
    elif RSSI_value_selection=="RSSI_Right":
        X["RSSI"] = X.RSSI_Right
    elif RSSI_value_selection=="Min":
        X["RSSI"] = X[['RSSI_Left','RSSI_Right']].min(axis=1).values
    elif RSSI_value_selection=="Max":
        X["RSSI"] = X[['RSSI_Left','RSSI_Right']].max(axis=1).values
    else: 
        X["RSSI"] = np.ceil(X[['RSSI_Left','RSSI_Right']].mean(axis=1).values).astype('int')

    X, features_name = extract_features(X)
    X.drop('Device_ID', axis=1, inplace=True)
    return X[features_name]


#########################

DEFAULT_TRUE_LIST = [True] * 7
DEFAULT_TRUE_DICT = {
    'RSSI': [True, False, False, False, True, True, True],
    'RSSI_diffs': [True, True, True, False, True, True, True],
    'RSSI_diffs_abs': [False, False, True, True, True, False, True],
    'RSSI_median_dist': [True, False, True, True, True, False, True]
}

def train_extract_features(data, bases=None):
    """
    Extract features from data.
    :param data: Dataset of time windows.
    :param bases: Dictionary with values of bool lists of size 7 and keys of the names of the vectors to extract
    features from
    :return: new dataset with extracted features, training feature name list
    """

    if bases is None:
        bases = DEFAULT_TRUE_DICT

    features_name = []
    data['RSSI_diffs'] = data.RSSI.apply(lambda x: x[1:] - x[:-1])
    data['RSSI_diffs_abs'] = data.RSSI.apply(lambda x: abs(x[1:] - x[:-1]))
    data['RSSI_median_dist'] = data.RSSI.apply(lambda x: abs(x - np.median(x)))

    data, features_name = extract_list_feats('RSSI', data, features_name, base=bases['RSSI'])
    data, features_name = extract_list_feats('RSSI_diffs', data, features_name, base=bases['RSSI_diffs'])
    data, features_name = extract_list_feats('RSSI_diffs_abs', data, features_name, base=bases['RSSI_diffs_abs'])
    data, features_name = extract_list_feats('RSSI_median_dist', data, features_name, base=bases['RSSI_median_dist'])

    data['max_count_same_value_RSSI'] = data.RSSI.apply(lambda x: np.max(np.unique(x, return_counts=True)[1]))
    features_name += ['max_count_same_value_RSSI']

    data['RSSI_peaks'] = data.RSSI.apply(lambda x: len(find_peaks(x)[0]))
    features_name += ['RSSI_peaks']

    data['RSSI_diffs_peaks'] = data.RSSI_diffs.apply(lambda x: len(find_peaks(x)[0]))
    features_name += ['RSSI_diffs_peaks']

    data['peak_ratio_diffs_RSSI'] = data.apply(
        lambda x: x['RSSI_diffs_peaks'] / x['RSSI_peaks'] if x['RSSI_peaks'] > 0 else 0, axis=1)
    features_name += ['peak_ratio_diffs_RSSI']

    data['RSSI_values_count'] = data.RSSI.apply(lambda x: len(np.unique(x)))
    features_name += ['RSSI_values_count']

    return data, features_name


def train_window(full_signal: np.ndarray, size: int = 360, stride: int = 360):
    """
    Take a long vector of signals and creates time windows of size "size" and stride of size "stride"
    :param full_signal: the signal to make time windows from
    :param size: size of each time window
    :param stride: time window stride (step size). When window size <= stride it's mean that there is not overlap between the windows.
    :return: time windows of the signal
    """
    return np.lib.stride_tricks.sliding_window_view(full_signal, size)[0::stride]

def train_make_data(X, y, window_size: int = 360, stride: int = 360):
    """
    Make data for training a model: making windows, adding metadata information to the time windows dataframe, removing
    windows with change in Num_People
    :param X: the data.
    :param y: the labels
    :param window_size: size of each time window
    :param stride: time window stride (step size). When window size <= stride it's mean that there is not overlap between the windows.
    :return: windowed RSSI DataFrame , labels dataframe
    """
    
    X['Num_People'] = y
    multi_vals = X.groupby(['Device_ID']).apply(lambda x: x.nunique() == 1).all()
    single_vals = list(multi_vals[multi_vals].index)
    multi_vals = list(multi_vals[~multi_vals].index)
    windows_df = X.groupby(['Device_ID']).RSSI.apply(
        lambda x: train_window(x.values, window_size, stride)).explode().to_frame().reset_index()
    for col in (multi_vals + single_vals):
        windows_df[col] = X.groupby(['Device_ID'])[col].apply(
            lambda x: train_window(x.values, window_size, stride)).explode().reset_index(drop=True).values
    for col in single_vals:
        windows_df[col] = windows_df[col].apply(lambda x: x[0])
    
    df = windows_df
    df['change'] = df.Num_People.apply(lambda x: (len(np.unique(x)) > 1))
    dfx = df[~df['change']]
    df = dfx.copy()
    df.Num_People = df.Num_People.apply(lambda x: x[0])
    df.drop(columns='change', inplace=True)
    return df.drop(columns='Num_People'), df.Num_People


def train_create_features(data, window_size, stride):
    """
    Feature engineering: 
    :param data: the data
    :param window_size: size of each time window
    :param stride: time window stride (step size). When window size <= stride it's mean that there is not overlap between the windows.
    :return: full dataset (with extracted features), train set y
    """

    X, y = data.drop(columns='Num_People'), data['Num_People']
    X, y = train_make_data(X, y, window_size=window_size, stride=stride)
    X_features, train_feat = train_extract_features(X.copy())
    train_feat.append('Device_ID')
    X_features = X_features[train_feat]
    return X_features, y, X


def train_pre_data(data, RSSI_value_selection, window_size, stride):
    """
    Full preprocessing of the data - train_x, train_y split, feature extraction,
    remove data that is smaller than the selected size window, etc.
    :param data: the row data.
    :param RSSI_value_selection: Which signal values to use.
    :param window_size: size of each time window
    :param stride: time window stride (step size). When window size <= stride it's mean that there is not overlap between the windows.
    :return: train set x (with extracted features per window), train set y
    """
    if RSSI_value_selection=="RSSI_Left":
        data["RSSI"] = data.RSSI_Left
    elif RSSI_value_selection=="RSSI_Right":
        data["RSSI"] = data.RSSI_Right
    elif RSSI_value_selection=="Min":
        data["RSSI"] = data[['RSSI_Left','RSSI_Right']].min(axis=1).values
    elif RSSI_value_selection=="Max":
        data["RSSI"] = data[['RSSI_Left','RSSI_Right']].max(axis=1).values
    else: 
        data["RSSI"] = np.ceil(data[['RSSI_Left','RSSI_Right']].mean(axis=1).values).astype('int')

    data.drop(['Room_Num'], axis=1, inplace=True)
    data.dropna(subset = ["Num_People"], inplace=True)

    for dev_id in list(set(data.Device_ID)):
        sub_dev_id = data.loc[data.Device_ID == dev_id]
        if len(sub_dev_id) < window_size:
            data = data[data.Device_ID != dev_id]
    train_x, train_y, raw_x = train_create_features(data, window_size, stride)
    train_x= train_x.reset_index(drop = True)
    train_y= train_y.reset_index(drop = True)
    train_x.drop('Device_ID', axis=1, inplace=True)
    return train_x, train_y, raw_x
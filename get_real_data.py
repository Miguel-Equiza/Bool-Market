import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from synthetic_data import *
from preprocessing import *
def get_real_X_y(noise=False, pattern_list = ["rising_wedge", "falling_wedge", "double_top", "double_bottom"]):

    X1=[]
    y1=[]
    X2=[]
    y2=[]
    X3=[]
    y3=[]
    X4=[]
    y4=[]
    for pattern in pattern_list:
        directory_path = f'data/patterns/{pattern}'
        if pattern == pattern_list[0]:
            for root, dirs, files in os.walk(directory_path):
                for filename in files:
                    df = pd.read_csv(f"{directory_path}/{filename}")
                    X1.append(np.array(df[["Open", "High", "Low", "Close"]]))
                    y1.append(tuple(df.iloc[0][["Start Date", "End Date", "Pattern"]]))
        if pattern == pattern_list[1]:
            for root, dirs, files in os.walk(directory_path):
                for filename in files:
                    df = pd.read_csv(f"{directory_path}/{filename}")
                    X2.append(np.array(df[["Open", "High", "Low", "Close"]]))
                    y2.append(tuple(df.iloc[0][["Start Date", "End Date", "Pattern"]]))
        if pattern == pattern_list[2]:
            for root, dirs, files in os.walk(directory_path):
                for filename in files:
                    df = pd.read_csv(f"{directory_path}/{filename}")
                    X3.append(np.array(df[["Open", "High", "Low", "Close"]]))
                    y3.append(tuple(df.iloc[0][["Start Date", "End Date", "Pattern"]]))
        if pattern == pattern_list[3]:
            for root, dirs, files in os.walk(directory_path):
                for filename in files:
                    df = pd.read_csv(f"{directory_path}/{filename}")
                    X4.append(np.array(df[["Open", "High", "Low", "Close"]]))
                    y4.append(tuple(df.iloc[0][["Start Date", "End Date", "Pattern"]]))

    if noise:
        for i in range(len(X1)):
            pat = {"0":X2, "1":X3,"2":X4}
            p = np.random.randint(0,3)
            p = str(p)
            num = np.random.randint(0,len(pat[p]))
            X1.append(pat[p][num])
            y1.append([-1,-1,0])

        for i in range(len(X2)):
            pat = {"0":X1, "1":X3,"2":X4}
            p = np.random.randint(0,3)
            p = str(p)
            num = np.random.randint(0,len(pat[p]))
            X2.append(pat[p][num])
            y2.append([-1,-1,0])

        for i in range(len(X3)):
            pat = {"0":X1, "1":X2,"2":X4}
            p = np.random.randint(0,3)
            p = str(p)
            num = np.random.randint(0,len(pat[p]))
            X3.append(pat[p][num])
            y3.append([-1,-1,0])

        for i in range(len(X4)):
            pat = {"0":X1, "1":X2,"2":X3}
            p = np.random.randint(0,3)
            p = str(p)
            num = np.random.randint(0,len(pat[p]))
            X4.append(pat[p][num])
            y4.append([-1,-1,0])


    return X1, y1, X2, y2, X3, y3, X4, y4

def get_up_down(patterns=["uptrend","downtrend"]):
    X1=[]
    y1=[]
    X2=[]
    y2=[]
    for pattern in patterns:
        directory_path = f'data/patterns/{pattern}'
        if pattern == "uptrend":
            for root, dirs, files in os.walk(directory_path):
                for filename in files:
                    df = pd.read_csv(f"data/patterns/uptrend/{filename}")
                    X1.append(np.array(df[["Open", "High", "Low", "Close"]]))
                    y1.append(tuple(df.iloc[0][["Start Date", "End Date", "Pattern"]]))
        if pattern == "downtrend":
            for root, dirs, files in os.walk(directory_path):
                for filename in files:
                    df = pd.read_csv(f"data/patterns/downtrend/{filename}")
                    X2.append(np.array(df[["Open", "High", "Low", "Close"]]))
                    y2.append(tuple(df.iloc[0][["Start Date", "End Date", "Pattern"]]))

    return X1, y1, X2, y2

def data_augmentation(Xs: list[np.ndarray], ys: list[tuple[int, int, int]], noise = False, n=1):
    """Creates more patterns based on the list of given ones, optionally with pattern."""
    assert len(Xs) == len(ys)
    new_Xs, new_ys = [], []

    for X, y in zip(Xs, ys):
        size = len(X)
        start, end, pattern = y

        start_margin, end_margin = round(start * (2/3)), round((size - end )*(2/3))
        if start_margin > 4 and end_margin > 4:

            for _ in range(n):
                margin_left = np.random.randint(4, start_margin)
                margin_right = np.random.randint(4, end_margin)
                new_Xs.append(X[start - margin_left : end + margin_right])
                new_ys.append((margin_left, margin_left + end - start, pattern)) # margin_left, margin_right + end - start, pattern

        if noise: # then add also the ones without pattern from start and end
            if start > 9: # then the size is enough to have a time-series before
                new_Xs.append(X[:start])
                new_ys.append((-1,-1,0))

            if size - end > 9:
                new_Xs.append(X[end:])
                new_ys.append((-1,-1,0))

    return new_Xs, new_ys

def upside_down(Xs, ys):
    assert len(Xs) == len(ys)
    new_Xs, new_ys = [], []

    # pat={
    #     "rising_wedge":1,
    #     "falling_wedge":2,
    #     "double_top":3,
    #     "double_bottom":4
    # }
    inverse_patterns = {
        1: 2,
        2: 1,
        3: 4,
        4: 3,
        5:6,
        6:5,
        7:8,
        8:7
    }
    for X, y in zip(Xs, ys):
        start, end, pattern = y
        if pattern != 0:
            new_pattern = inverse_patterns[pattern]
            l = X - np.max(X[start:end])
            o = -l
            new_Xs.append(o)
            new_ys.append((start, end, new_pattern))

    return new_Xs, new_ys

def augmentation(X1, y1, X2, y2, X3, y3, X4, y4, noise=True):
    x_1, y_1 = data_augmentation(X1, y1, noise=noise)
    x_2, y_2 = data_augmentation(X2, y2, noise=noise)
    x_3, y_3 = data_augmentation(X3, y3, noise=noise)
    x_4, y_4 = data_augmentation(X4, y4, noise=noise)

    X_2, Y_2 = upside_down(x_1, y_1)
    X_1, Y_1 = upside_down(x_2, y_2)
    X_4, Y_4 = upside_down(x_3, y_3)
    X_3, Y_3 = upside_down(x_4, y_4)

    X1 = x_1+X_1
    X2 = x_2+X_2
    X3 = x_3+X_3
    X4 = x_4+X_4

    y1 = y_1 + Y_1
    y2 = y_2 + Y_2
    y3 = y_3 + Y_3
    y4 = y_4 + Y_4

    return X1, y1, X2, y2, X3, y3, X4, y4


def augmentate(X1, y1, noise=True, n=1):
    x_1, y_1 = data_augmentation(X1, y1, noise=noise, n=n)
    X_2, Y_2 = upside_down(x_1, y_1)
    X = x_1 + X_2
    y = y_1 + Y_2

    return X, y

def prim_separation(X_train, y_train):
    X_train_sec, y_train_sec, X_train_prim, y_train_prim = [],[],[],[]
    for X, y in zip(X_train, y_train):
        if y[2] > 4 :
            X_train_sec.append(X)
            y_train_sec.append(y)
        else:
            X_train_prim.append(X)
            y_train_prim.append(y)
    return X_train_sec, y_train_sec, X_train_prim, y_train_prim


def get_data(synth=True):
    X1, y1, X2, y2, X3, y3, X4, y4 = get_real_X_y(noise=False, pattern_list = ["rising_wedge", "falling_wedge", "double_top", "double_bottom"])
    X5, y5, X6, y6, X7, y7, X8, y8 = get_real_X_y(noise=False, pattern_list = ["ascending_triangle","descending_triangle","h&s_top","h&s_bottom"])
    if synth:

        X_ris_wedg, y_ris_wedg, X_fal_wedg, y_fal_wedg, X_d_top, y_d_top, X_d_bottom, y_d_bottom = get_X_y(noise=False, general=False)
        X_asc_tri, y_asc_tri, X_desc_tri, y_desc_tri, X_hs_top, y_hs_top, X_hs_bottom, y_hs_bottom = get_sec_X_y(noise=False, general=False)

        X = X1 + X2 + X3 + X4 + X_ris_wedg + X_fal_wedg + X_d_top + X_d_bottom + X5 + X6 + X7+ X8 + X_asc_tri + X_desc_tri + X_hs_top + X_hs_bottom
        y = y1 + y2 + y3 + y4 + y_ris_wedg + y_fal_wedg + y_d_top + y_d_bottom + y5 + y6 + y7 + y8 + y_asc_tri + y_desc_tri + y_hs_top + y_hs_bottom

    else:
        X = X1 + X2 + X3 + X4 + X5 + X6 + X7+ X8
        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    X_train_sec, y_train_sec, X_train_prim, y_train_prim = prim_separation(X_train, y_train)
    X_val_sec, y_val_sec, X_val_prim, y_val_prim = prim_separation(X_val, y_val)
    X_test_sec, y_test_sec, X_test_prim, y_test_prim = prim_separation(X_test, y_test)

    X_train_prim, y_train_prim = augmentate(X_train_prim, y_train_prim, noise=True, n=1)
    X_test_prim, y_test_prim = augmentate(X_test_prim, y_test_prim, noise=True, n=1)
    X_val_prim, y_val_prim = augmentate(X_val_prim, y_val_prim, noise=True, n=1)

    X_train_sec, y_train_sec = augmentate(X_train_sec, y_train_sec, noise=True, n=10)
    X_test_sec, y_test_sec = augmentate(X_test_sec, y_test_sec, noise=True, n=10)
    X_val_sec, y_val_sec = augmentate(X_val_sec, y_val_sec, noise=True, n=10)

    X_train = X_train_prim + X_train_sec
    X_test = X_test_prim + X_test_sec
    X_val = X_val_prim + X_val_sec

    y_train = y_train_prim + y_train_sec
    y_test = y_test_prim + y_test_sec
    y_val = y_val_prim + y_val_sec

    X_train_preprocessed, y_train_p1, y_train_p2, y_train_dates = preprocess(X_train, y_train)
    X_test_preprocessed, y_test_p1, y_test_p2, y_test_dates = preprocess(X_test, y_test)
    X_val_preprocessed, y_val_p1, y_val_p2, y_val_dates = preprocess(X_val, y_val)

    return X_train_preprocessed, y_train_p1, y_train_p2, y_train_dates, X_test_preprocessed, y_test_p1, y_test_p2, y_test_dates, X_val_preprocessed, y_val_p1, y_val_p2, y_val_dates


def get_single_data(X_train_preprocessed, y_train_p, y_train_dates, X_test_preprocessed, y_test_p, y_test_dates, pattern = ["rising_wedge", "falling_wedge", "double_bottom", "double_top"]):

    mapping = {
            "rising_wedge": [0, 0, 0, 1, 0],
            "falling_wedge": [0, 0, 1, 0, 0],
            "double_top": [0, 1, 0, 0, 0],
            "double_bottom": [1, 0, 0, 0, 0]
        }
    ohe = mapping[pattern]



    X_t, y_d, y_p = [], [], []
    for pattern, X, dates in zip(np.array(y_test_p), X_test_preprocessed, y_test_dates):
        if (((pattern == ohe).sum()) == 5):

            y_p.append(1)
            X_t.append(X)
            y_d.append(dates)

        if (((pattern == [0, 0, 0, 0, 1]).sum()) == 5):

            y_p.append(0)
            X_t.append(X)
            y_d.append(dates)

    X_test_preprocessed = tf.convert_to_tensor(X_t, np.float32)
    y_test_p = tf.convert_to_tensor(y_p, np.int16)
    y_test_dates = tf.convert_to_tensor(y_d, np.int16)

    X_t, y_d, y_p = [], [], []
    for pattern, X, dates in zip(np.array(y_train_p), X_train_preprocessed, y_train_dates):
        if ((pattern == ohe).sum()) == 5:
            y_p.append(1)
            X_t.append(X)
            y_d.append(dates)

        if (((pattern == [0, 0, 0, 0, 1]).sum()) == 5):

            y_p.append(0)
            X_t.append(X)
            y_d.append(dates)

    X_train_preprocessed = tf.convert_to_tensor(X_t, np.float32)
    y_train_p = tf.convert_to_tensor(y_p, np.int16)
    y_train_dates = tf.convert_to_tensor(y_d, np.int16)


    return X_train_preprocessed, y_train_p, y_train_dates, X_test_preprocessed, y_test_p, y_test_dates

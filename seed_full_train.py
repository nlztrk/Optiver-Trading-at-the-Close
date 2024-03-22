#!/usr/bin/env python
# coding: utf-8

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb
import numpy as np
import numba
import pandas as pd
import warnings
from tqdm.auto import tqdm
from itertools import combinations, product

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

whole_df = pd.read_csv("../comp_files/train.csv")

def tidy_up_df(hdf,
              target_dateid=False,
              target_second=False):
    
    # Chronological sorting
    hdf = hdf.sort_values(by=["stock_id",
                            "date_id",
                            "seconds_in_bucket"],
                        ascending=True).reset_index(drop=True)

    # Adding missing rows by reindexing
    all_stock_ids = np.arange(0., 200., 1.)
    all_seconds = np.arange(0., 550., 10.)
    all_date_ids = list(hdf["date_id"].unique())
    ideal_group_ids = list(product(all_stock_ids, all_date_ids, all_seconds))

    hdf = hdf.set_index(["stock_id","date_id","seconds_in_bucket"])
    hdf = hdf.reindex(ideal_group_ids).reset_index()
    
    # Chronological sorting on imputed data
    hdf = hdf.sort_values(by=["stock_id",
                            "date_id",
                            "seconds_in_bucket"],
                        ascending=True).reset_index(drop=True)
    
    # Creating row_id for new added rows, will be used on prediction merging @ inference phase
    # row_id = date_id _ seconds_in_bucket _ stock_id
    missing_row_idxs = hdf.row_id.isna()

    hdf.loc[missing_row_idxs, "row_id"] = hdf.loc[missing_row_idxs, "date_id"].astype(int).astype(str) + "_" +\
                            hdf.loc[missing_row_idxs, "seconds_in_bucket"].astype(int).astype(str) + "_" +\
                            hdf.loc[missing_row_idxs, "stock_id"].astype(int).astype(str)    
    
    # Will be used on inference stage
    if target_dateid or target_second:
        hdf = hdf[(hdf["date_id"] == target_dateid) &\
                  (hdf["seconds_in_bucket"] == target_second)].reset_index(drop=True)
    
    return hdf


whole_df = tidy_up_df(whole_df)

stock_weights = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
    0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
    0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
    0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
    0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
    0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
    0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
    0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
    0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
    0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
    0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
    0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
    0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
    0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
    0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
    0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
]

drop_cols = [
    'date_id',
    'row_id',
    'time_id',
    'target',
    
    'fold1',
    'fold2',
    'fold3',
    'fold4',
    
    "last_matched_size",
    'currently_scored'
                 ]

nonpct_cols = drop_cols + [
    'stock_id',
    'seconds_in_bucket',
    'matched_diff_date',
    'day_of_week'
                 ]

original_f32_cols = ['stock_id', 'date_id', 'seconds_in_bucket', 'imbalance_size',
       'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
       'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price',
       'ask_size', 'wap', 'target']


def run_anil_pipe(df, kaggle_mode=False):

    if kaggle_mode:
        df[original_f32_cols] = df[original_f32_cols].astype(np.float32)
    else:
        for col in original_f32_cols:
            df[col] = df[col].astype(np.float32)
            
    # Index features with raw prices
    df["stock_weight"] = df["stock_id"].astype(int).apply(lambda x: stock_weights[x]).astype(np.float32)
    
    df["weighted_wap"] = (df["stock_weight"] * df["wap"]).astype(np.float32)
    df = df.merge(df.groupby(["date_id", "seconds_in_bucket"])["weighted_wap"].sum()\
                              .rename("index_coeff").astype(np.float32).reset_index(),
                             how="left",
                             on=["date_id", "seconds_in_bucket"])    

    df['wap_index_ratio'] = df.eval("wap / index_coeff").astype(np.float32)
    df['wap_index_diff'] = df.eval("wap - index_coeff").astype(np.float32)
    
    #

    
    # Price Normalization
    for col in df.columns:
        if ("price" in col or col in ["wap", "weighted_wap"]) and col != "reference_price":
            df[col] /= df["reference_price"]
        
    df["imbalance"] = (df["imbalance_size"] * df["imbalance_buy_sell_flag"]).astype(np.float32)
    df["day_of_week"] = (df["date_id"] % 5).astype(np.float32)
    
    # Intra-day second-specific Features
    intraday_cols = ['imbalance_size',
       'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
       'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price',
       'ask_size', 'wap', 'target']
    
    for day_offset in [1]:
        df[[f"{col}_dayoffset{day_offset}_shifted" for col in intraday_cols]] =\
            df.groupby(["stock_id"])[intraday_cols].shift(day_offset*55).astype(np.float32)
        
    # Last Day Comparisons
    stock_date_idxs = df.groupby(["stock_id", "date_id"])["matched_size"].last().index
    #
    current_day_start_match_size = df.groupby(["stock_id", "date_id"])["matched_size"].first()\
            .rename("start_of_the_day_matched_size").reset_index()
    df = df.merge(current_day_start_match_size, how="left", on=["stock_id", "date_id"])
    del current_day_start_match_size
    #                                                
    current_day_start_imbalance = df.groupby(["stock_id", "date_id"])["imbalance"].first()\
            .rename("current_day_start_imbalance").reset_index()
    df = df.merge(current_day_start_imbalance, how="left", on=["stock_id", "date_id"]) 
    del current_day_start_imbalance
    #                                               
    last_day_ending_match_size = pd.DataFrame(df.groupby(["stock_id", "date_id"])["matched_size"].last()\
            .reset_index().groupby(["stock_id"])["matched_size"].shift()\
            .rename("last_day_ending_match_size")).set_index(stock_date_idxs).reset_index()
    df = df.merge(last_day_ending_match_size, how="left", on=["stock_id", "date_id"])
    del last_day_ending_match_size
    #    
    last_day_ending_imbalance = pd.DataFrame(df.groupby(["stock_id", "date_id"])["imbalance"].last()\
            .reset_index().groupby(["stock_id"])["imbalance"].shift()\
            .rename("last_day_ending_imbalance")).set_index(stock_date_idxs).reset_index()
    df = df.merge(last_day_ending_imbalance, how="left", on=["stock_id", "date_id"])
    del last_day_ending_imbalance
    
    del stock_date_idxs
    
    #
    df["matched_diff_date"] = df["start_of_the_day_matched_size"] - df["last_day_ending_match_size"]
    df["matched_ratio_date"] = df["start_of_the_day_matched_size"] / df["last_day_ending_match_size"]
    
    df["current_matched_diff_date"] = df["matched_size"] - df["last_day_ending_match_size"]
    df["current_matched_ratio_date"] = df["matched_size"] / df["last_day_ending_match_size"]
    
    df["imbalance_diff_date"] = df["current_day_start_imbalance"] - df["last_day_ending_imbalance"]
    df["imbalance_ratio_date"] = df["current_day_start_imbalance"] / df["last_day_ending_imbalance"]
    
    df["current_imbalance_diff_date"] = df["imbalance"] - df["last_day_ending_imbalance"]
    df["current_imbalance_ratio_date"] = df["imbalance"] / df["last_day_ending_imbalance"]                     
    

    ###########
    # https://www.kaggle.com/code/nyanpn/1st-place-public-2nd-place-solution
    
    for col in ["wap", "ask_price", "bid_price"]:
        df[f'log_{col}'] = np.log(df[col])
        df[f'log_return_{col}'] = df.groupby(["stock_id", "date_id"])[f'log_{col}']\
            .diff().astype(np.float32).values
        df.drop(columns=[f'log_{col}'], axis=1, inplace=True)
        
        df[f'log_return_{col}_sq'] = df[f'log_return_{col}'] ** 2
        df[f'log_return_{col}_realized_volatility'] = df.groupby(["stock_id", "date_id"])[f'log_return_{col}_sq'].expanding().\
                sum().astype(np.float32).values
        df[f'log_return_{col}_realized_volatility'] = np.sqrt(df[f'log_return_{col}_realized_volatility'])
        df.drop(columns=[f'log_return_{col}_sq'], axis=1, inplace=True)
    
    ###########
    
    
    df['bid_price_diff_ask_price'] = df.eval("bid_price - ask_price").astype(np.float32)
    df['bid_price_sum_ask_price'] = df.eval("bid_price + ask_price").astype(np.float32)
    df['bid_size_sum_ask_size'] = df.eval("bid_size + ask_size").astype(np.float32)

    df["volume"] = df.eval("ask_size + bid_size").astype(np.float32)
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2").astype(np.float32)
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)").astype(np.float32)
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)").astype(np.float32)
    df["size_imbalance"] = df.eval("bid_size / ask_size").astype(np.float32)

    df["wap_1st_nominator"] = df.eval("ask_size * bid_price").astype(np.float32)
    df["wap_2nd_nominator"] = df.eval("ask_price * bid_size").astype(np.float32)
    df["wap_whole_nominator"] = df.eval("wap_1st_nominator + wap_2nd_nominator").astype(np.float32)

    df["imbalance_momentum"] = (df.groupby(["stock_id", "date_id"])['imbalance_size'].diff(periods=1) \
                                / df['matched_size']).astype(np.float32)
    
    df["price_spread"] = (df["ask_price"] - df["bid_price"]).astype(np.float32)
    df["spread_intensity"] = df.groupby(["stock_id", "date_id"])['price_spread'].diff().astype(np.float32)
    df['price_pressure'] = (df['imbalance_size'] * (df['ask_price'] - df['bid_price'])).astype(np.float32)
    df['market_urgency'] = (df['price_spread'] * df['liquidity_imbalance']).astype(np.float32)
    df['depth_pressure'] = ((df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])).astype(np.float32)

    df['spread_depth_ratio'] = ((df['ask_price'] - df['bid_price']) / (df['bid_size'] + df['ask_size'])).astype(np.float32)
    df['mid_price_movement'] = df['mid_price'].diff(periods=5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).astype(np.float32)

    df['mid_price*volume'] = (df['mid_price_movement'] * df['volume']).astype(np.float32)
    df['harmonic_imbalance'] = df.eval('2 / ((1 / bid_size) + (1 / ask_size))').astype(np.float32)

    # Linear combinations of both sizes and prices
    df['reverse_wap'] = df.eval('((bid_price * bid_size) + (ask_price * ask_size)) / (bid_size + ask_size)').astype(np.float32)

    # Liquidity Features
    df['bid_ask_spread_imbalance_interaction'] = (df['bid_price_diff_ask_price'] * df['imbalance_size']).astype(np.float32)
    df['bid_ask_spread_percentage'] = ((df['bid_price_diff_ask_price']) / ((df['bid_price_sum_ask_price']) / 2)).astype(np.float32)

    # WAP Rolling Features
    for period1 in [3,6,12]:
        df[f'wap_ema_{period1}'] = df.groupby(["stock_id", "date_id"])['wap'].rolling(period1).mean().astype(np.float32).values
        df[f'price_momentum_{period1}'] = (df['wap'] / df[f'wap_ema_{period1}'] - 1).astype(np.float32)
        df[f'wap_min_period{period1}'] = df.groupby(["stock_id", "date_id"])['wap'].rolling(period1).min().astype(np.float32).values
        df[f'wap_max_period{period1}'] = df.groupby(["stock_id", "date_id"])['wap'].rolling(period1).max().astype(np.float32).values
    
    df[f'wap_expanding_min'] = df.groupby(["stock_id", "date_id"])['wap'].expanding().min().astype(np.float32).values
    df[f'wap_expanding_max'] = df.groupby(["stock_id", "date_id"])['wap'].expanding().max().astype(np.float32).values


    # Price to Bid-Ask Ratio
    df['price_to_bid_ask_diff_ratio'] = (df['wap'] / df['bid_price_diff_ask_price']).astype(np.float32)

    # Spread-to-Volume Ratio
    df['spread_to_volume_ratio'] = (df['bid_price_diff_ask_price'] / df['matched_size']).astype(np.float32)

    # order_book_depth-to-Volume Ratio
    df['order_book_depth_to_matched_ratio'] = (df['bid_size_sum_ask_size'] / df['matched_size']).astype(np.float32)

    # Average Trade Size
    df['average_trade_size'] = (df['matched_size'] / (df['seconds_in_bucket'] + 1)).astype(np.float32)

    # Relative Spread
    df['relative_spread'] = (df['bid_price_diff_ask_price'] / df['wap']).astype(np.float32)

    
    df = df.drop(columns=["imbalance_size", "imbalance_buy_sell_flag"], axis=1)

        
    ################
    
    
    pctcols = list(set(df.columns).difference(nonpct_cols +\
                                              [f"{col}_dayoffset1_shifted" for col in intraday_cols] +\
                                             ["last_day_ending_match_size", "last_day_ending_imbalance"]))
    
    df[["allstocks_" + "_".join(elm) + "_pctrank" for elm in pctcols]] =\
        df.groupby(["date_id", "seconds_in_bucket"])[list(pctcols)]\
            .rank(pct=True).astype(np.float32)

    ################

    pctcols = list(set(df.columns).difference(nonpct_cols +\
                                             ["last_day_ending_match_size", "last_day_ending_imbalance"] +\
                                              [col for col in df.columns if "dayoffset" in col] +\
                                              [col for col in df.columns if "expanding" in col] +\
                                              [col for col in df.columns if "ema_" in col] +\
                                              [col for col in df.columns if "_period" in col] +\
                                              [col for col in df.columns if "momentum_" in col]
                                             )
                  )
    
    # Nested PCT Fest 2023
    for period1 in tqdm([1, 3, 6, 12, 18]):
        shifted_cols = df.groupby(["stock_id", "date_id"])[pctcols].shift(period1)
        df[[f"{pctcol}_period{period1}_diff" for pctcol in pctcols]] = (df[pctcols] - shifted_cols).astype(np.float32)
        del shifted_cols

    numerical_cols = list(set(df.columns).difference(["row_id", "currently_scored"]))
    
    if kaggle_mode:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df[numerical_cols] = df[numerical_cols].astype(np.float32)
    else:
        for col in numerical_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].astype(np.float32)
    
    return df


whole_df = run_anil_pipe(whole_df, kaggle_mode=False)
whole_df.dropna(subset=["target"], inplace=True)
whole_df.reset_index(inplace=True, drop=True)

features = sorted(list(set(whole_df.columns).difference(drop_cols)))
target = "target"

SEED = 1

collist = list(features.copy())
np.random.seed(SEED)  
np.random.shuffle(collist) 

xgb_param = {
  "learning_rate": 0.01,
  "colsample_bytree": 0.8,
  "colsample_bylevel": 0.9,
  "colsample_bynode": 0.9,
  "subsample": 0.7,
  "max_depth": 9,
  "gamma": 0.01,
  "min_child_weight": 250,
  "lambda": 0.01,
  "alpha": 0,
  "seed": SEED,
  "objective": 'reg:absoluteerror',
  "eval_metric": 'mae',
  "tree_method": 'auto',
  "device": 'cuda',
}

d_train = xgb.DMatrix(whole_df[collist], whole_df[target])
model = xgb.train(xgb_param, d_train, num_boost_round=4500)
model.save_model(f'model_full_seed{SEED}.json')
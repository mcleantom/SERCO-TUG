# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:10:44 2021

@author: Rastko
"""

import pandas as pd
import numpy as np


def read_csv(filepath):
    df = pd.read_csv(filepath, usecols=["imei", "dataloggertime", "longitude",
                                        "latitude", "speedoverground",
                                        "revolutions", "enginetorque",
                                        "fuelrate", "motor_id", "percentload"])
    df.dropna(subset = ["imei"])
    df["dataloggertime"] = pd.to_datetime(df["dataloggertime"])
    df = df.sort_values(by=["dataloggertime"])
    df = df.set_index(["dataloggertime"])
    
    engines = split_engines(df)
    for i in engines:
        engines[i] = calc_torque(engines[i])
        engines[i] = calc_power(engines[i])
    
    return df, engines


def split_engines(df):
    return dict(tuple(df.groupby('motor_id')))


def split_days(df):
    return dict(tuple(df.groupby([df.index.year, df.index.month, df.index.day])))

def split_seconds(df):
    return df.groupby([df.index.year,
                       df.index.month,
                       df.index.day,
                       df.index.minute,
                       df.index.second])

def calc_torque(df):
    """
    Fit the engine RPM to a torque curve
    """
    df["torque"] = np.nan

    mask = df["revolutions"] < 1025

    rpm = df.loc[mask, "revolutions"]
    df.loc[mask, "torque"] = (0.00001698*rpm**3 -
                              0.0273*rpm**2 +
                              14.99*rpm + 1276)

    mask = df["revolutions"] >= 1025

    rpm = df.loc[mask, "revolutions"]
    df.loc[mask, "torque"] = (-0.0000005388274*rpm**4 +
                             +0.003072432*rpm**3 -
                             6.551486*rpm**2 +
                             6186.143*rpm - 2171670)
    return df


def calc_power(df):
    df["power"] = (df["percentload"]/100)*2*np.pi*(df["revolutions"]/60)*(df["torque"]/1000)
    return df


def sum_engine_powers(engines):
    combined = engines[203].join(engines[204], how="outer", rsuffix="_2")

file = "Data\engine_data_week_8.csv"
data, engines = read_csv(file)

# engines = split_engines(data)
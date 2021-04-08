# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:10:44 2021

@author: Rastko
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap

engine_curves = pd.read_excel("Data/engine_curve.xlsx")

def read_csv(filepath):
    df = pd.read_csv(filepath, usecols=["imei", "dataloggertime", "longitude",
                                        "latitude", "speedoverground",
                                        "revolutions", "enginetorque",
                                        "fuelrate", "bus_id", "percentload"])
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
    return dict(tuple(df.groupby('bus_id')))


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
    combined = engines[1].join(engines[2], how="outer", rsuffix="_2")
    combined["total_power"] = combined["power"] + combined["power_2"]
    combined["ave_rpm"] = (combined["revolutions"] + combined["revolutions_2"])/2
    return combined


def plot_power_vs_rpm(df):
    fig, ax = plt.subplots()
    cols = engine_curves.columns[1:]
    for col in cols:
        plt.plot(engine_curves["RPM"], engine_curves[col], label="Engine Power "+str(col)+"%")
    
    ax.plot(df["revolutions"], df["total_power"], 'o', markersize=0.5, color="blue")
    ax.set_xlim(600, 1600)
    ax.set_ylim(0, 2500)
    ax.set_xlabel("Engine RPM")
    ax.set_ylabel("Total Power [kW]")
    ax.legend(loc="upper left")
    ax.grid()


def plot_power_vs_sog(df):
    fig, ax = plt.subplots()
    # cols = engine_curves.columns[1:]
    # for col in cols
    
    ax.plot(df["speedoverground"], df["total_power"], 'o', markersize=0.5, color="blue")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 2500)
    ax.set_xlabel("SOG [kts]")
    ax.set_ylabel("Total Power [kW]")
    ax.legend(loc="upper left")

# def draw_map(df):
    
#     lat_mid = df["lateral"].mean()
#     long_mean = df["longitudinal"].mean()
#     width = df["lateral"].max()-df["lateral"].min()
#     height = df["longitudinal"].max()-df["longitudinal"].min()
    
#     m = Basemap(width=width, height=height, projection='lcc', resolution='c',
#                 lat_1 = lat_mid-width/2, lat_2 = lat_mid-width/2,
#                 lon)

file = "Data\engine_data_week_8.csv"
data, engines = read_csv(file)
combined = sum_engine_powers(engines)
# engines = split_engines(data)
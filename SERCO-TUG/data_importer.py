# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:10:44 2021

@author: Rastko
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

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
    combined["speedoverground"] = combined["speedoverground"].replace(to_replace=0, method='ffill')
    return combined


def plot_power_vs_rpm(df):
    fig, ax = plt.subplots(dpi=512) 
    ax.plot(df["ave_rpm"], df["total_power"], 'o', markersize=0.5, color="blue")
    cols = engine_curves.columns[1:]
    for col in cols:
        plt.plot(engine_curves["RPM"], engine_curves[col], label="Engine Power "+str(col)+"%")
   

    ax.set_xlim(600, 1600)
    ax.set_ylim(0, 2500)
    ax.set_xlabel("Engine RPM")
    ax.set_ylabel("Total Power [kW]")
    ax.legend(loc="upper left")
    ax.grid()


def plot_power_vs_sog(df):
    fig, ax = plt.subplots(dpi=512)
    
    cols = engine_curves.columns[1:]
    for col in cols:
        ax.plot([0, 7], [engine_curves[col][0]]*2, label="Engine Power "+str(col)+"%")
    
    ax.plot(df["speedoverground"], df["total_power"], 'o', markersize=0.5, color="blue")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 2500)
    ax.set_xlabel("SOG [kts]")
    ax.set_ylabel("Total Power [kW]")
    ax.legend(loc="upper left")
    ax.grid()


def plot_day(df):
    
    start = df.index[0].to_pydatetime().date()
    end = start# + datetime.timedelta(days=1)
    start = datetime.datetime(start.year, start.month, start.day, 6)
    end = datetime.datetime(end.year, end.month, end.day, 20)
    fig, ax1 = plt.subplots(dpi=512)
    
    ax1.set_ylabel("Total Power [kW], RPM")
    df["total_power"].plot(ax=ax1, x_compat=True, color="black", label="Total Power", linewidth=0.8)
    df["ave_rpm"].plot(ax=ax1, x_compat=True, color="blue", label="RPM", linewidth=0.8)

    ax1.set_xlim([start, end])
    ax1.grid()
    ax1.legend(loc="upper left")

    xticks = pd.date_range(start, end, freq='H')
    ax1.set_xticklabels([x.strftime('%H') for x in xticks])
    ax1.xaxis.set_tick_params(rotation=00)

    ax2 = ax1.twinx()
    ax2.set_ylabel('SOG [knots]')
    
    df[df["speedoverground"]>0]["speedoverground"].plot(ax=ax2, x_compat=True, color="red", label="SOG", linewidth=0.8)
    
    ax2.legend(loc="upper right")
    
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    
    ax1.set_xlabel("Hour of the day [hrs]")
    ax1.set_ylim([0, 2500])
    ax2.set_ylim([0, 10])


def split_trips(df):
    g = (df.index.to_series().diff().dt.seconds > 1).cumsum()
    trips = dict(tuple(df.groupby(g)))
    for trip in trips:
        trips[trip] = trips[trip][trips[trip]["imei"].notna()]
    return trips


def plot_trips(df):
    
    start = df.index[0].to_pydatetime().date()
    end = start# + datetime.timedelta(days=1)
    start = datetime.datetime(start.year, start.month, start.day, 6)
    end = datetime.datetime(end.year, end.month, end.day, 20)
    
    fig, ax1 = plt.subplots(dpi=512)
    ax1.set_ylabel("Total Power [kW], RPM")
    ax1.set_xlim([start, end])
    
    trips = split_trips(df)
    for trip in trips:
        trips[trip]["ave_rpm"].plot(ax=ax1, x_compat=True, label="Trip " + str(trip))

    xticks = pd.date_range(start, end, freq='H')
    ax1.set_xticklabels([x.strftime('%H') for x in xticks])
    ax1.xaxis.set_tick_params(rotation=00)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    
    ax1.grid()
    ax1.set_xlabel("Hour of the day [hrs]")
    ax1.set_ylim([0, 2500])
    ax1.legend()
    
file = "Data\engine_data_2021_02.csv"
data, engines = read_csv(file)
combined = sum_engine_powers(engines)
plot_power_vs_rpm(combined)
plot_power_vs_sog(combined)
days = split_days(combined)

for day in days:
    plot_day(days[day])
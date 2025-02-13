from enum import Enum
from typing import Tuple, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from time import localtime
import requests
import click

NOAA_NCEI_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


class Station(Enum):
    """NOAA Station IDs

    To find more station IDs for more places, go to www.ncei.noaa.gov
    """

    MADISON_WI = "USW00014837"  # Dane county regional airport
    BOSTON_MA = "USW00014739"
    BOULDER_CO = "USC00050848"
    HALF_MOON_BAY_CA = "USC00043714"

    @classmethod
    def from_str(cls, string: str):
        s = string.replace(", ", " ")
        s = s.replace(" ", "_")
        s = s.replace(",", "_")
        s = s.upper()
        for station in cls:
            if s == station.name:
                return station
        raise ValueError(f"Cannot translate string {string} into Station enum.")

    def to_str(self):
        if self == Station.MADISON_WI:
            return "Madison, WI"
        if self == Station.BOSTON_MA:
            return "Boston, MA"
        if self == Station.BOULDER_CO:
            return "Boulder, CO"
        if self == Station.HALF_MOON_BAY_CA:
            return "Half Moon Bay, CA"


def get_ncei_data(station: Station, start_date="1959-08-01", end_date=None) -> pd.DataFrame:
    api_url = NOAA_NCEI_URL
    if end_date is None:
        end_date = str(date.today())
    parameters = [
        "?dataset=daily-summaries",
        "stations={}".format(station.value),
        "dataTypes=SNOW,SNWD",
        "startDate={}".format(start_date),
        "endDate={}".format(end_date),
        "format=json",
        "includeAttributes=false",
        "includeStationName=true",
    ]
    request_url = api_url + "&".join(parameters)
    result = requests.get(request_url).json()
    if isinstance(result, dict):
        if "errorCode" in result.keys():
            raise requests.exceptions.RequestException(result)
    return pd.DataFrame(result)


def season(row):
    if row["Day"] < 200:
        return row["YEAR"] - 1
    else:
        return row["YEAR"]


def season_day(row):
    if row["Day"] < 200:
        return row["Day"]
    else:
        if row["leap_year"]:
            return row["Day"] - 366
        else:
            return row["Day"] - 365


def process_data(df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["DATE"] = pd.to_datetime(df.DATE)
    df["YEAR"] = pd.to_datetime(df.DATE).dt.year

    for c in ["SNOW", "SNWD"]:
        df[c] = df[c].astype(float)

    # create groupby dataframe
    gb = df.drop(["STATION"], axis=1).groupby("DATE").mean(numeric_only=True)
    gb["Day"] = pd.to_datetime(gb.index.values).dayofyear
    gb["leap_year"] = gb.index.is_leap_year
    gb["Season"] = gb.apply(season, axis=1)
    gb["New_Day"] = gb.apply(season_day, axis=1)
    gb["cumulative_snow"] = (
        gb.sort_values(["New_Day"]).groupby(["Season"]).SNOW.cumsum()
    )
    # create percentile dataframes
    pctl_snowfall = pd.DataFrame()
    pctl_snowfall["pct_05"] = gb.groupby(["New_Day"]).cumulative_snow.quantile(0.05)
    pctl_snowfall["pct_25"] = gb.groupby(["New_Day"]).cumulative_snow.quantile(0.25)
    pctl_snowfall["pct_75"] = gb.groupby(["New_Day"]).cumulative_snow.quantile(0.75)
    pctl_snowfall["pct_95"] = gb.groupby(["New_Day"]).cumulative_snow.quantile(0.95)
    pctl_snowfall["Min"] = gb.groupby(["New_Day"]).cumulative_snow.min()
    pctl_snowfall["Max"] = gb.groupby(["New_Day"]).cumulative_snow.max()
    pctl_snowfall["Mean"] = gb.groupby(["New_Day"]).cumulative_snow.mean()

    pctl_snowdepth = pd.DataFrame()
    pctl_snowdepth["pct_05"] = gb.groupby(["New_Day"]).SNWD.quantile(0.05)
    pctl_snowdepth["pct_25"] = gb.groupby(["New_Day"]).SNWD.quantile(0.25)
    pctl_snowdepth["pct_75"] = gb.groupby(["New_Day"]).SNWD.quantile(0.75)
    pctl_snowdepth["pct_95"] = gb.groupby(["New_Day"]).SNWD.quantile(0.95)
    pctl_snowdepth["Min"] = gb.groupby(["New_Day"]).SNWD.min()
    pctl_snowdepth["Max"] = gb.groupby(["New_Day"]).SNWD.max()
    pctl_snowdepth["Mean"] = gb.groupby(["New_Day"]).SNWD.mean()

    return gb, pctl_snowfall, pctl_snowdepth


def make_snowfall_figure(station: Station, gb: pd.DataFrame, pctl: pd.DataFrame) -> plt.Figure:
    current_season = season(
        {
            "Day": localtime().tm_yday,
            "YEAR": localtime().tm_year,
        }
    )
    legend: List[str] = []
    f = plt.figure()
    sns.set_theme(style="darkgrid")

    plt.plot(pctl.index, pctl["Min"], color="k", alpha=0.2)
    legend.append("Minimum cumulative snowfall")
    plt.plot(pctl.index, pctl["Max"], color="k", alpha=0.2)
    legend.append("Maximum cumulative snowfall")

    plt.plot(pctl.index, pctl.pct_05, color="tab:blue", alpha=0.2)
    legend.append("5th Percentile")
    plt.plot(pctl.index, pctl.pct_95, color="tab:blue", alpha=0.2)
    legend.append("95th Percentile")

    plt.fill_between(
        pctl.index,
        pctl.pct_25,
        pctl.pct_75,
        color="tab:blue",
        alpha=0.2,
    )
    legend.append("Middle two quartiles")
    plt.plot(pctl.index, pctl.Mean, color="tab:blue")
    legend.append("Average")

    # p = sns.lineplot(x="New_Day", y="cumulative_snow", data=gb, ci=None)

    plt.plot(
        gb[gb.Season == current_season - 2].New_Day,
        gb[gb.Season == current_season - 2].cumulative_snow,
        color="g",
    )
    legend.append("{}-{} Season".format(current_season - 2, current_season - 1))
    plt.plot(
        gb[gb.Season == current_season - 1].New_Day,
        gb[gb.Season == current_season - 1].cumulative_snow,
        color="r",
    )
    legend.append("{}-{} Season".format(current_season - 1, current_season))
    plt.plot(
        gb[gb.Season == current_season].New_Day,
        gb[gb.Season == current_season].cumulative_snow,
        color="k",
    )
    legend.append("{}-{} Season".format(current_season, current_season + 1))

    plt.ylabel("Cumulative Snow (mm)")
    plt.xlabel("Day of Season (day 1 = Jan 1st)")
    plt.title("Cumulative Snowfall in {}".format(station.to_str()))
    plt.legend(legend)

    return f


def make_snowdepth_figure(station: Station, gb: pd.DataFrame, pctl: pd.DataFrame) -> plt.Figure:
    current_season = season({"Day": localtime().tm_yday, "YEAR": localtime().tm_year})

    f = plt.figure()
    sns.set_theme(style="darkgrid")

    plt.plot(pctl.index, pctl["Min"], color="k", alpha=0.2)
    plt.plot(pctl.index, pctl["Max"], color="k", alpha=0.2)

    plt.plot(pctl.index, pctl.pct_05, color="tab:blue", alpha=0.2)
    plt.plot(pctl.index, pctl.pct_95, color="tab:blue", alpha=0.2)
    plt.fill_between(
        pctl.index,
        pctl.pct_25,
        pctl.pct_75,
        color="tab:blue",
        alpha=0.2,
    )
    plt.plot(pctl.index, pctl.Mean, color="tab:blue")

    # p = sns.lineplot(x="New_Day", y="cumulative_snow", data=gb, ci=None)

    plt.plot(
        gb[gb.Season == current_season - 2].New_Day,
        gb[gb.Season == current_season - 2].SNWD,
        color="g",
    )
    plt.plot(
        gb[gb.Season == current_season - 1].New_Day,
        gb[gb.Season == current_season - 1].SNWD,
        color="r",
    )
    plt.plot(
        gb[gb.Season == current_season].New_Day,
        gb[gb.Season == current_season].SNWD,
        color="k",
    )

    plt.ylabel("Snow Depth (mm)")
    plt.xlabel("Day of Season (day 1 = Jan 1st)")
    plt.title("Snow Depth in {}".format(station.to_str()))
    plt.legend(
        [
            "Mininum Snow Depth",
            "Maximum Snow Depth",
            "5th Percentile",
            "95th Percentile",
            "Middle two quartiles",
            "Average",
            "{}-{} Season".format(current_season - 2, current_season - 1),
            "{}-{} Season".format(current_season - 1, current_season),
            "{}-{} Season".format(current_season, current_season + 1),
        ]
    )

    return f

def make_day_of_the_week_snow_figure(station: Station, gb: pd.DataFrame) -> plt.Figure:

    gb["Day_name"] = pd.to_datetime(gb.index.values).day_name()
    day_of_week_gb = gb[gb.Season == 2022].groupby(by=["Season", "Day_name"]).sum(numeric_only=True)
    ax = day_of_week_gb.plot.bar("Day_name", "SNOW")
    return ax

@click.command()
@click.option(
    "--location", default="MADISON_WI", help="Location for historical snowfall graphs"
)
def main(location):
    savefile_name = "{}-{}-" + "{}.png".format(date.today())
    station = Station.from_str(location)
    data_raw = get_ncei_data(station)
    data = data_raw.copy()

    gb, pctl_snowfall, pctl_snow_depth = process_data(data)

    snowfall_figure = make_snowfall_figure(station, gb, pctl_snowfall)
    snowfall_figure.savefig(savefile_name.format(station.name, "snowfall"))

    snow_depth_figure = make_snowdepth_figure(station, gb, pctl_snow_depth)
    snow_depth_figure.savefig(savefile_name.format(station.name, "snowdepth"))

    dow_figure = make_day_of_the_week_snow_figure(station, gb)
    dow_figure.savefig(savefile_name.format(station.name, "dow_snowfall"))


if __name__ == "__main__":
    main()

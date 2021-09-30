import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from time import localtime
import requests
import click

stanion_ids = {
    "Half Moon Bay, CA": "USC00043714",
    "Boston, MA": "USW00014739",
    "Madison, WI": "USW00014837",
}


def get_ncei_data(station_id, start_date="1939-10-01", end_date=None):
    api_url = "https://www.ncei.noaa.gov/access/services/data/v1"
    if end_date is None:
        end_date = str(date.today())
    parameters = [
        "?dataset=daily-summaries",
        "stations={}".format(station_id),
        "dataTypes=SNOW,SNWD",
        "startDate={}".format(start_date),
        "endDate={}".format(end_date),
        "format=json",
        "includeAttributes=false",
        "includeStationName=true",
    ]
    request_url = api_url + "&".join(parameters)
    return pd.DataFrame(requests.get(request_url).json())


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


def process_data(df):
    df["DATE"] = pd.to_datetime(df.DATE)
    df["YEAR"] = pd.DatetimeIndex(df.DATE).year

    for c in ["SNOW", "SNWD"]:
        df[c] = df[c].astype(float)

    # create groupby dataframe
    gb = df.drop(["STATION"], axis=1).groupby("DATE").mean()
    gb["Day"] = pd.DatetimeIndex(gb.index).dayofyear
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


def make_snowfall_figure(location, gb, pctl) -> plt.Figure:
    current_season = season(
        {
         "Day": localtime().tm_yday,
         "YEAR": localtime().tm_year,
        }
    )

    f = plt.figure()
    sns.set(style="darkgrid")

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
        gb[gb.Season == current_season - 2].cumulative_snow,
        color="g",
    )
    plt.plot(
        gb[gb.Season == current_season - 1].New_Day,
        gb[gb.Season == current_season - 1].cumulative_snow,
        color="r",
    )
    plt.plot(
        gb[gb.Season == current_season].New_Day,
        gb[gb.Season == current_season].cumulative_snow,
        color="k",
    )

    plt.ylabel("Cumulative Snow (in.)")
    plt.xlabel("Day of Season (day 1 = Jan 1st)")
    plt.title("Cumulative Snowfall in {}".format(location))
    plt.legend(
        [
            "Mininum cumulative snowfall",
            "Maximum cumulative snowfall",
            "5th Percentile",
            "95th Percentile",
            "Average",
            "{}-{} Season".format(current_season - 2, current_season - 1),
            "{}-{} Season".format(current_season - 1, current_season),
            "{}-{} Season".format(current_season, current_season + 1),
            "Middle two quartiles",
        ]
    )

    return f


def make_snowdepth_figure(location, gb, pctl) -> plt.Figure:
    current_season = season(
        {
         "Day": localtime().tm_yday,
         "YEAR": localtime().tm_year
        }
    )

    f = plt.figure()
    sns.set(style="darkgrid")

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

    plt.ylabel("Snow Depth (in.)")
    plt.xlabel("Day of Season (day 1 = Jan 1st)")
    plt.title("Snow Depth in {}".format(location))
    plt.legend(
        [
            "Mininum Snow Depth",
            "Maximum Snow Depth",
            "5th Percentile",
            "95th Percentile",
            "Average",
            "{}-{} Season".format(current_season - 2, current_season - 1),
            "{}-{} Season".format(current_season - 1, current_season),
            "{}-{} Season".format(current_season, current_season + 1),
            "Middle two quartiles",
        ]
    )

    return f


@click.command()
@click.option(
    "--location",
    default="Madison, WI",
    help="Location for historical snowfall graphs"
)
def main(location):
    savefile_name = "{}-{}-" + "{}.png".format(date.today())
    station = stanion_ids[location]
    data_raw = get_ncei_data(station)
    data = data_raw.copy()

    gb, pctl_snowfall, pctl_snow_depth = process_data(data)

    snowfall_figure = make_snowfall_figure(location, gb, pctl_snowfall)
    snowfall_figure.savefig(savefile_name.format(location, "snowfall"))

    snow_depth_figure = make_snowdepth_figure(location, gb, pctl_snow_depth)
    snow_depth_figure.savefig(savefile_name.format(location, "snowdepth"))


if __name__ == "__main__":
    main()

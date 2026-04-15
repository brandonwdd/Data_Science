"""Aggregate trip pickups into hourly demand."""

import pandas as pd


def hourly_demand(trips: pd.DataFrame) -> pd.DataFrame:
    """
    Count trips per clock hour. Reindex to a complete hourly range;
    hours with no pickups get demand 0.
    """
    t = trips.copy()
    t["hour"] = t["pickup_datetime"].dt.floor("h")
    counts = t.groupby("hour", as_index=False).size()
    counts = counts.rename(columns={"size": "demand"})

    full_index = pd.date_range(
        counts["hour"].min(),
        counts["hour"].max(),
        freq="h",
    )
    series = (
        counts.set_index("hour")
        .reindex(full_index, fill_value=0)
        .rename_axis("hour")
        .reset_index()
    )
    series.columns = ["hour", "demand"]
    return series

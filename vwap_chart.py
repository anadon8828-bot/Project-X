"""Bar-based VWAP estimate. Intraday resets by exchange-local session date."""
import numpy as np
import pandas as pd


def with_vwap(data, intraday=True, timezone='Asia/Tokyo'):
    result=data.sort_index().copy()
    index=pd.DatetimeIndex(result.index)
    if intraday:
        local=index.tz_localize(timezone) if index.tz is None else index.tz_convert(timezone)
        groups=local.date
    else:
        groups=np.zeros(len(result),dtype=int)
    volume=pd.to_numeric(result.Volume,errors='coerce').where(lambda s:s>=0)
    typical=(result.High+result.Low+result.Close)/3
    result['VWAP']=(typical*volume).groupby(groups).cumsum()/volume.groupby(groups).cumsum().replace(0,np.nan)
    return result

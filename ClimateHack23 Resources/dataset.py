import pandas as pd
from datetime import datetime, time, timedelta
from torch.utils.data import IterableDataset

class ChallengeDataset(IterableDataset):
    def __init__(self, pv, hrv, site_locations, hrv_buffer = 1, start_date = "2020-7-1", end_date = "2020-7-30", sites=None):
        self.pv = pv
        self.hrv = hrv
        self.hrv_buffer = hrv_buffer
        self._site_locations = site_locations
        self._sites = sites if sites else list(site_locations["hrv"].keys())
        self.start_date = list(map(int, start_date.split("-")))
        self.end_date= list(map(int, end_date.split("-")))

    def _get_image_times(self):
        min_date = datetime(self.start_date[0], self.start_date[1], self.start_date[2])
        max_date = datetime(self.end_date[0], self.end_date[1], self.end_date[2])
        
        start_time = time(8)
        end_time = time(17)

        date = min_date 
        while date <= max_date: 
            current_time = datetime.combine(date, start_time)
            while current_time.time() < end_time:
                if current_time:
                    yield current_time

                current_time += timedelta(minutes=60)

            date += timedelta(days=1)

    def __iter__(self):
        for time in self._get_image_times():

            time_ids = pd.date_range(start=time + timedelta(hours=1),
                                     end=time + timedelta(hours=1) + timedelta(minutes=55),
                                     freq='5min')
            time_ids = time_ids.strftime('%Y-%m-%dT%H:%M:%S').tolist()

            # 1 hour leading up to the predicton time        
            first_hour = slice(str(time), str(time + timedelta(minutes=55)))

            # PV power output in first hour
            pv_features = self.pv.xs(first_hour, drop_level=False)

            # PV power output in the horizon period
            pv_targets = self.pv.xs(
                slice(  # type: ignore
                    str(time + timedelta(hours=1)),
                    str(time + timedelta(hours= 1, minutes=55)),
                ),
                drop_level=False,
            )

           # hrv satellite images on first hour timestamps setting them up as an input feature
            hrv_data = self.hrv["data"].sel(time=first_hour).to_numpy()

            for site in self._sites:
                site_id = site

                try:
                    # Get solar PV features and targets, the site_targets is used to find the models loss
                    site_features = pv_features.xs(site, level=1).to_numpy().squeeze(-1) 
                    site_targets = pv_targets.xs(site, level=1).to_numpy().squeeze(-1)
                    assert site_features.shape == (12,) and site_targets.shape == (12,)
                 
                    # Get a HRV crop centred on the site over the previous hour
                    x, y = self._site_locations["hrv"][site]
                    hrv_features = hrv_data[:, y - self.hrv_buffer  : y + self.hrv_buffer ,
                                             x - self.hrv_buffer  : x + self.hrv_buffer , 0]
                    assert hrv_features.shape == (12, 2*self.hrv_buffer, 2*self.hrv_buffer)

                except:
                    continue

                yield time_ids, site_id, site_features, hrv_features, site_targets
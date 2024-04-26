import pandas as pd
from datetime import datetime, time, timedelta
from torch.utils.data import IterableDataset

class Dataset(IterableDataset):
    def __init__(self, pv, hrv, site_locations,
                 is_incident= False, start_date = "2020-7-1", end_date = "2020-7-30",
                   crop_size = 1, horizon = 1, sites=None):
        self.pv = pv
        self.hrv = hrv
        self._site_locations = site_locations
        self.is_incident = is_incident
        self._sites = sites if sites else list(site_locations["hrv"].keys())#This gets the individual site ids which are stored as the dict's keys
        self.start_date = list(map(int, start_date.split("-")))
        self.end_date= list(map(int, end_date.split("-")))
        self.crop_size = crop_size
        self.horizon = horizon

    def _get_image_times(self):#This function starts at the minimum date in the set and iterates up to the highest date, this is done as the data set is large and due to the nature of the parquette and xarray
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
            
            # generate time ids for predictions to be analysedm after training
            time_ids = pd.date_range(start=time + timedelta(hours=1),
                                     end=time + timedelta(hours=self.horizon)+timedelta(minutes=55),
                                     freq='5min')
            time_ids = time_ids.strftime('%Y-%m-%dT%H:%M:%S').tolist()  

            # 1 hour leading up to the predicton time        
            first_hour = slice(str(time), str(time + timedelta(minutes=55)))

            # PV power output in first hour
            if self.is_incident:
                pv_features = self.pv.xs(first_hour, drop_level=False)[["power", "angle_of_incidence_radians"]]
            else:  
                pv_features = self.pv.xs(first_hour, drop_level=False)

            # PV power output in the next 48 hours
            pv_targets = self.pv.xs(
                slice(  # type: ignore
                    str(time + timedelta(hours=1)),
                    str(time + timedelta(hours= self.horizon, minutes=55)),
                ),
                drop_level=False,
            )
            if self.is_incident:
                pv_targets = pv_targets["power"]

           # hrv satellite images on first hour timestamps setting them up as an input feature
            hrv_data = self.hrv["data"].sel(time=first_hour).to_numpy()

            for site in self._sites:
                site_id = site
                try:
                    # Get solar PV features and targets, the site_targets is used to find the models loss
                    site_features = pv_features.xs(site, level=1).to_numpy().squeeze(-1) # gets the pixel based location of the pv site and then uses this to make predictions based on the individual sites
                    site_targets = pv_targets.xs(site, level=1).to_numpy().squeeze(-1)
                    
                    if self.is_incident:
                        assert site_features.shape == (12,2) and site_targets.shape == (12*self.horizon,)
                    else:
                        assert site_features.shape == (12,) and site_targets.shape == (12*self.horizon,)
                 
                    # Get a (2*crop_size + 1)^2 HRV crop centred on the site over the previous hour
                    x, y = self._site_locations["hrv"][site]
                    hrv_features = hrv_data[:, y - self.crop_size  : y + self.crop_size ,
                                             x - self.crop_size  : x + self.crop_size , 0]
                    assert hrv_features.shape == (12, self.crop_size, self.crop_size)

                except:
                    continue

                yield  time_ids, site_id, site_features, hrv_features, site_targets
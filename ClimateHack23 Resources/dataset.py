import pandas as pd
from datetime import datetime, time, timedelta
from torch.utils.data import DataLoader, IterableDataset

class Dataset(IterableDataset):
    def __init__(self, pv, hrv, site_locations,
                  start_date = "2020-7-1", end_date = "2020-7-30",
                   crop_size = 1, horizon = 1, sites=None):
        self.pv = pv
        self.hrv = hrv
        self._site_locations = site_locations
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
            time_ids = pd.date_range(start=time + timedelta(hours=1),
                                     end=time + timedelta(hours=self.horizon)+timedelta(minutes=55),
                                     freq='5min')
            time_ids = time_ids.strftime('%Y-%m-%dT%H:%M:%S').tolist()        
            first_hour = slice(str(time), str(time + timedelta(minutes=55)))#gets the time and then uses this to select the corresponding time from the pv set  

            pv_features = self.pv.xs(first_hour, drop_level=False)  # this gets the pv yield of the current timestamp selected earlier
            pv_targets = self.pv.xs(
                slice(  # type: ignore
                    str(time + timedelta(hours=1)),
                    str(time + timedelta(hours= self.horizon, minutes=55)),
                ),
                drop_level=False,
            )#pv targets defines the time span over which we are trying to make pv yield predictions
    
            hrv_data = self.hrv["data"].sel(time=first_hour).to_numpy()#gets the hrv satellite image that is associated with the first hour timestamp setting it up as an input feature

            for site in self._sites:
                site_id = site
                try:
                    # Get solar PV features and targets, the site_targets is used to find the models loss
                    site_features = pv_features.xs(site, level=1).to_numpy().squeeze(-1)#gets the pixel based location of the pv site and then uses this to make predictions based on the individual sites
                    site_targets = pv_targets.xs(site, level=1).to_numpy().squeeze(-1)
                    assert site_features.shape == (12,) and site_targets.shape == (12,)#compresses the data from N dimensions to 12 and 48 respectively
                 
                    # Get a (2*crop_size + 1)^2 HRV crop centred on the site over the previous hour
                    x, y = self._site_locations["hrv"][site]#gets the location of the site based on the pv sites pixel level location
                    hrv_features = hrv_data[:, y - self.crop_size  : y + self.crop_size , x - self.crop_size  : x + self.crop_size , 0]
                    assert hrv_features.shape == (12, self.crop_size, self.crop_size)# crops the image to be around the site
                    #assert is used to force the dimensions of the extracted site level image to be the same

                except:
                    continue

                yield  time_ids, site_id, site_features, hrv_features, site_targets
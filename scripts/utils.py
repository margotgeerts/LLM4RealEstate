import pandas as pd
import numpy as np
import os
import json
from functools import partial
import geopy
import geocoder
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.neighbors import NearestNeighbors
import re

# add the repo to the path
import sys
repo_dir = '/LLM4RealEstate'
os.chdir(repo_dir)
sys.path.append(repo_dir)

n = {0: 'first',
     1: 'second',
     2: 'third',
     3: 'fourth',
     4: 'fifth',
     5: 'sixth',
     6: 'seventh',
     7: 'eighth',
     8: 'ninth',
     9: 'tenth'}

class Prompt:
    def __init__(self, dataset):
        self.dataset = dataset
        self.geolocator = Nominatim(user_agent="anon")
        geocode_reverse = partial(self.geolocator.reverse, language='en')
        self.geocode = RateLimiter(geocode_reverse, min_delay_seconds=1)
        self.df = pd.read_csv(f'data/processed/{dataset}.csv', index_col=0)
        self.train = self.df[self.df['set']=='train']
        self.val = self.df[self.df['set']=='val']
        self.test = self.df[self.df['set']=='test']
        self.max_price = np.round(np.exp(self.train['log_price']).max())
        self.median_price = np.round(np.exp(self.train['log_price']).median())
        self.min_price = np.round(np.exp(self.train['log_price']).min())
        self.task_definition = json.load(open('config/task_definition.json'))[self.dataset]
        self.prompt_templates = json.load(open('config/prompt_template.json'))
        self.context = json.load(open('config/context.json'))[self.dataset]
        self.features = json.load(open('data/features.json'))[self.dataset]
        self.system_prompt = "You are a real estate expert."
        self.currency = json.load(open('config/currency.json'))[self.dataset]

    def reload_data(self):
        self.df = pd.read_csv(f'data/processed/{self.dataset}.csv', index_col=0)
        self.train = self.df[self.df['set']=='train']
        self.val = self.df[self.df['set']=='val']
        self.test = self.df[self.df['set']=='test']
    
    def get_address(self, row):
        if 'address' not in row or pd.isna(row['address']):
            coords = f"{row['y_geo']}, {row['x_geo']}"
            location = self.geocode(coords)
            address = location.raw['address']
            formatted_address = []
            for key in address:
                if "-" in key or "number" in key or "code" in key:
                    continue
                formatted_address.append(address[key])

            row['address'] = ", ".join(formatted_address)

            # save the address
            self.df.loc[row.name , 'address'] = row['address']
            self.df.to_csv(f'data/processed/{self.dataset}.csv')
            self.reload_data()

        return row['address']

    def get_examples(self, row, n_examples, example_selection='geo'):
        if example_selection == 'mixed':
            return self.get_examples_mixed(row, n_examples)
        elif example_selection == 'geo':
            features = ['x_geo', 'y_geo']
            metric = 'haversine'
        else:
            features = self.features['hedonic']
            metric = 'cosine'
        nn = NearestNeighbors(n_neighbors=n_examples, metric=metric)
        nn.fit(self.train[features].values)
        distances, indices = nn.kneighbors(row[features].values.reshape(1, -1))
        return self.train.iloc[indices[0]]

    def get_examples_mixed(self, row, n_examples):
        # first check if a file with the examples exists
        example_file = f'data/{self.dataset}_test_10_mixed_examples.json'
        if os.path.exists(example_file):
            with open(example_file) as f:
                examples = json.load(f)
            return self.train.iloc[examples[str(int(row.name))]]

        nn_geo = NearestNeighbors(n_neighbors=n_examples, metric='haversine')
        nn_geo.fit(self.train[['x_geo', 'y_geo']].values)
        nn_hedonic = NearestNeighbors(n_neighbors=n_examples, metric='cosine')
        nn_hedonic.fit(self.train[self.features['hedonic']].values)
        _, indices_geo = nn_geo.kneighbors(row[['x_geo', 'y_geo']].values.reshape(1, -1))
        _, indices_hedonic = nn_hedonic.kneighbors(row[self.features['hedonic']].values.reshape(1, -1))
        indices_geo = indices_geo[0]
        indices_hedonic = indices_hedonic[0]
        
        half_n = n_examples // 2
        indices = pd.unique(np.concatenate([indices_geo[:half_n], indices_hedonic[:half_n]]))
        cnt = 0
        while len(indices) < n_examples:
            cnt += 1
            if cnt > 10:
                print("Could not find enough examples")
                break
            # calculate the difference between indices and indices_geo
            indices_geo_left = np.setdiff1d(indices_geo, indices, assume_unique=True)
            if len(indices) % 2 == 1:
                # get the next index from the geo neighbors that is not yet in indices
                new_indices = np.array(indices_geo_left[:1])
            else:
                new_indices = np.array([indices_geo_left[0], np.setdiff1d(indices_hedonic, indices, assume_unique=True)[0]])
            indices = pd.unique(np.concatenate([indices, new_indices]))
        return self.train.iloc[indices]

    
    def get_price_prompt(self, row, context=False, n_examples=0, example_selection='geo'):
        price_prompt = self.task_definition + " "
        if context:
            year_month = row.transaction_date[:7]
            if year_month in self.context:
                price_prompt += self.prompt_templates['context'].format(context_report=self.context[year_month])
        if n_examples > 0:
            price_prompt += self.prompt_templates['example_start']
            examples = self.get_examples(row, n_examples, example_selection)
            for i, (j, example) in enumerate(examples.iterrows()):
                hedonic_features = format_hedonic_features(example, self.dataset)
                price_prompt += self.prompt_templates['example'][self.dataset].format(
                    nth = n[i],
                    x_geo = example['x_geo'],
                    y_geo = example['y_geo'],
                    address = self.get_address(example),
                    **hedonic_features,
                    transaction_date = example['transaction_date'],
                    price = np.round(np.exp(example['log_price']) if 'price' not in example else example['price'])
                )
        hedonic_features = format_hedonic_features(row, self.dataset)
        price_prompt += self.prompt_templates['target'][self.dataset].format(
            x_geo = row['x_geo'],
            y_geo = row['y_geo'],
            address = self.get_address(row),
            **hedonic_features,
            transaction_date = row['transaction_date'],
            min_price = self.min_price,
            max_price = self.max_price,
            median_price = self.median_price
        )

        return price_prompt


def format_hedonic_features(row, dataset):
    if dataset == 'KC':
        return {
        "sqft_lot" : row['sqft_lot'],
        "sqft_living" : row['sqft_living'],
        "sqft_above" : row['sqft_above'],
        "bedrooms" : row['bedrooms'],
        "bathrooms" : row['bathrooms'],
        "floors" : row['floors'],
        "waterfront" : "" if row['waterfront']==1. else "not",
        "view" : row['view'],
        "condition" : row['condition'],
        "grade" : row['grade'],
        "yr_built" : row['yr_built']}
    elif dataset == 'beijing':
        return {
            "square": row['square'],
            "livingRoom": row['livingRoom'],
            "drawingRoom" : row['drawingRoom'],
            "kitchen" : row['kitchen'],
            "bathRoom" : row['bathRoom'],
            "floor" : row['floor'],
            "buildingType" : row['buildingType'],
            "constructionTime": row['constructionTime'],
            "renovationCondition" : row['renovationCondition'],
            "buildingStructure" : row['buildingStructure'],
            "ladderRatio": row['ladderRatio'],
            "elevator" : "an" if row['elevator']==1. else "no",
            "fiveYearsProperty" : "" if row['fiveYearsProperty']==1. else "not",
            "subway" : "a" if row['subway']==1. else "no"
        }
    elif dataset == 'barcelona':
        return {
            "CONSTRUCTEDAREA": row['CONSTRUCTEDAREA'],
            "ROOMNUMBER": row['ROOMNUMBER'],
            "BATHNUMBER": row['BATHNUMBER'],
            "HASTERRACE": "a" if row['HASTERRACE']==1. else "no",
            "HASLIFT": "a" if row['HASLIFT']==1. else "no",
            "HASAIRCONDITIONING": "" if row['HASAIRCONDITIONING']==1. else "no",
            "AMENITYID": row['AMENITYID'],
            "HASPARKINGSPACE": "a" if row['HASPARKINGSPACE']==1. else "no",
            "ISPARKINGSPACEINCLUDEDINPRICE": "" if row['ISPARKINGSPACEINCLUDEDINPRICE']==1. else "not ",
            "PARKINGSPACEPRICE": row['PARKINGSPACEPRICE'],
            "HASNORTHORIENTATION": "" if row['HASNORTHORIENTATION']==1. else "no ",
            "HASSOUTHORIENTATION": "" if row['HASSOUTHORIENTATION']==1. else "no ",
            "HASEASTORIENTATION": "" if row['HASEASTORIENTATION']==1. else "no ",
            "HASWESTORIENTATION": "" if row['HASWESTORIENTATION']==1. else "no ",
            "HASBOXROOM": "a" if row['HASBOXROOM']==1. else "no",
            "HASWARDROBE": "a" if row['HASWARDROBE']==1. else "no",
            "HASSWIMMINGPOOL": "a" if row['HASSWIMMINGPOOL']==1. else "no",
            "HASDOORMAN": "a" if row['HASDOORMAN']==1. else "no",
            "HASGARDEN": "a" if row['HASGARDEN']==1. else "no",
            "ISDUPLEX": "a" if row['ISDUPLEX']==1. else "not a",
            "ISSTUDIO": "a" if row['ISSTUDIO']==1. else "not a",
            "ISINTOPFLOOR": "" if row['ISINTOPFLOOR']==1. else "not ",
            "CONSTRUCTIONYEAR": row['CONSTRUCTIONYEAR'],
            "FLOORCLEAN": row['FLOORCLEAN'],
            "FLATLOCATIONID": row['FLATLOCATIONID'],
            "CADCONSTRUCTIONYEAR": row['CADCONSTRUCTIONYEAR'],
            "CADMAXBUILDINGFLOOR": row['CADMAXBUILDINGFLOOR'],
            "CADDWELLINGCOUNT": row['CADDWELLINGCOUNT'],
            "CADASTRALQUALITYID": row['CADASTRALQUALITYID'],
            "BUILTTYPEID_1": "" if row['BUILTTYPEID_1']==1. else "not",
            "BUILTTYPEID_2": "" if row['BUILTTYPEID_2']==1. else "not ",
            "BUILTTYPEID_3": "" if row['BUILTTYPEID_3']==1. else "not "}


def get_price_from_response(response, currency=None):
    response = response.lower()
    # if </think> is in the response, remove everything before it
    if "</think>" in response:
        response = response.split("</think>")[-1]
    if currency == "EUR":
        c = ["eur", "€", "euro", "euros"]
    elif currency == "USD":
        c = ["usd", "$", "dollar", "dollars"]
    elif currency == "CNY":
        c = ["cny", "¥", "yuan"]
    # find all occurrences of 'price EUR: number'
    price = re.findall(fr'price {c[0]}\'? ?:?=? ?(\d+,?\d+,?\d+\.?\d+)', response)
    if len(price) > 0:
        price = price[-1].replace(",", "")
        return float(price)
    else:
        # find all occurrences of 'price: number'
        price = re.findall(fr'price ?:?=? ?(\d+,?\d+,?\d+\.?\d+)', response)
        if len(price) > 0:
            price = price[-1].replace(",", "")
            return float(price)
        else:
            # find all occurrences of 'number {currency}'
            for cur in c:
                price = re.findall(fr'(\d+,?\d+,?\d+\.?\d+) ?{cur}', response)
                if len(price) > 0:
                    price = price[-1].replace(",", "")
                    return float(price)
            for cur in c:
                price = re.findall(fr'{cur} ?(\d+,?\d+,?\d+\.?\d+)', response)
                if len(price) > 0:
                    price = price[-1].replace(",", "")
                    return float(price)
    return None
                
        

    

def strip_word(word):
    chars_to_remove = [",", "$", "'", '"', "*"]
    for char in chars_to_remove:
        word = word.replace(char, "")
    return word.strip()



def get_interval_from_response(response):
    response = response.lower()    
    # if </think> is in the response, remove everything before it
    if "</think>" in response:
        response = response.split("</think>")[-1]
    
    try:
        interval = re.findall(r'(-?\d+,?\d+,?\d+\.?\d+) - (\d+,?\d+,?\d+\.?\d+)', response)
        min_price, max_price = interval[-1]
        min_price, max_price = min_price.replace(",", ""), max_price.replace(",", "")
        return float(min_price), float(max_price)
    # catch any error
    except (ValueError, IndexError):
        try: 
            min_price = re.findall(r'min_price ?\:?=? ?(-?\d+,?\d+,?\d+\.?\d+)', response)[-1]
            max_price = re.findall(r'max_price ?\:?=? ?(\d+,?\d+,?\d+\.?\d+)', response)[-1]
            min_price, max_price = min_price.replace(",", ""), max_price.replace(",", "")
            return float(min_price), float(max_price)
        except (ValueError, IndexError):
            return None, None


def get_features_from_response(response):
    # if </think> is in the response, remove everything before it
    if "</think>" in response:
        response = response.split("</think>")[-1]
    response = response.split(",")
    features = []
    for feature in response:
        features.append(feature.strip())
    return features
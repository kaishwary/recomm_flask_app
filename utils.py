import pandas as pd
import numpy as np

from annoy import AnnoyIndex

### CONFIG
# vector dimension
VECTOR_SIZE = 64
# metric to calculate vector similarity
METRIC='euclidean'
# Number of orders/baskets to pull similar to the requested
ORDER_RETURNS = 15
# Number of trees for queries. When making a query the more trees the easier it is to go down the right path.
TREES = 10
# Number of product recommendation as maximum
# NUMBER_OUTPUT_PRODUCTS = 10
# Sample size for the TSNE model and plot
TSNE_SIZE = 100
# Threshold for a minimum support
THRESHOLD_SUPPORT = 1e-3
# Threshold for the maximun number of products to bring
THRESHOLD_TOP = 10
# Threshold for distance, based on the quantile calculation of the basket distances
THRESHOLD_DISTANCE = 0.1
# Number of neighbours to choose from for daytime impact
# More number of neighbours => wider filter
DAYTIME_NEIGHBOURS = 5

def load_dataframes():
    products = pd.read_pickle('res/products.pkl')
    order_baskets = pd.read_pickle('res/order_baskets.pkl')
    df_daytime_basket = pd.read_pickle('res/df_daytime_basket.pkl')

    return products, order_baskets, df_daytime_basket

def load_annoy_objects():

    def load_obj(path):
        obj = AnnoyIndex(VECTOR_SIZE, metric=METRIC)
        obj.load(path)
        return obj

    p = load_obj('res/product.ann')
    u = load_obj('res/user.ann')
    b = load_obj('res/basket.ann')
    d = load_obj('res/daytime.ann')

    return p, u, b, d
    
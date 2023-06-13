"""
Annoy Index Loader

This code loads pre-trained Annoy indexes and provides utility functions for loading dataframes. It also defines various configuration parameters for vector size, metric, threshold values, and sample sizes.

The code performs the following tasks:
- Imports necessary modules and libraries for data processing and vector indexing.
- Defines configuration parameters for vector size, metric, thresholds, and sample sizes.
- Implements utility functions to load pre-trained Annoy indexes and dataframes.
- Loads the Annoy indexes for product, user, basket, and daytime vectors.

Utility Functions:
- load_dataframes: Loads the product, order_basket, and df_daytime_basket dataframes from pickle files.
- load_annoy_objects: Loads the pre-trained Annoy indexes for product, user, basket, and daytime vectors.

Configuration Parameters:
- VECTOR_SIZE: Dimension of the vectors.
- METRIC: Metric used to calculate vector similarity.
- ORDER_RETURNS: Number of orders/baskets to pull similar to the requested.
- TREES: Number of trees for queries.
- TSNE_SIZE: Sample size for the t-SNE model and plot.
- THRESHOLD_SUPPORT: Threshold for minimum support.
- THRESHOLD_TOP: Threshold for the maximum number of products to bring.
- THRESHOLD_DISTANCE: Threshold for distance, based on the quantile calculation of the basket distances.
- DAYTIME_NEIGHBOURS: Number of neighbours to choose from for daytime impact.

File name: utils.py
"""

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
    
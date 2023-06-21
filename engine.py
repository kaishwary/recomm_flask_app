"""
Recommendation Engine

This code implements a recommendation engine that generates product recommendations based on user preferences and input vectors. 
It uses various data processing and machine learning techniques, including clustering, filtering, and t-SNE visualization.

The code performs the following tasks:
- Imports necessary modules and libraries.
- Loads dataframes and pre-trained models.
- Implements functions for filtering and generating product recommendations based on user preferences and input vectors.
- Defines a class, RecommendationEngine, that provides an interface for generating predictions and visualizing results.
- Provides a t-SNE plot function to visualize product vectors in a 2D space.

The RecommendationEngine class provides the following methods:
- generatePredictions: Generates product recommendations based on user preferences, product lists, and a specified day and time.
- tSNEPlot: Generates a t-SNE plot to visualize product vectors in a 2D space, highlighting recommendations and input products.

File name: engine.py
"""

import plotly.express as px
from sklearn.manifold import TSNE
from art import *
import numpy as np
import pandas as pd
from efficient_apriori import apriori

import json
import plotly
import plotly.express as px
from sklearn.manifold import TSNE

from utils import *

try:
    # Load static files
    products, order_baskets = load_dataframes()
    # Load annoy objects
    p, u, b = load_annoy_objects()
except:
    print("Unable to load files, recreate files !")

PRODUCT_ID_MAP = dict(zip(products.product_name, products.product_id))

def rank_by_euclidean(df, vector):
    df['distance'] = df['vectors'].apply(lambda x: np.linalg.norm(x - vector))
    df = df.sort_values('distance', ascending=False)
    return df

def rank_by_dot_product(df, vector):
    df['dot_prod'] = df['vectors'].apply(lambda x: np.dot(x, vector))
    df = df.sort_values('dot_prod', ascending=False)
    return df

def compose_basket_by_cart(product_vector, input = None, n_items = 15, method='euclidean', n_neighbours = 100):
    order_list = b.get_nns_by_vector(product_vector, n_items)
    fpl = []
    for order in order_list:
        fpl = fpl + order_baskets[order]
    
    fpl = list(set(fpl))

    sel_df = pd.DataFrame({"product_id": fpl}).merge(products, on='product_id', how='inner')
    if method == 'euclidean':
        sel_df = rank_by_euclidean(sel_df, product_vector)
    else:  
        sel_df = rank_by_dot_product(sel_df, product_vector)
    if input is not None:
        sel_df = sel_df[~sel_df.product_id.isin(input)]
    sel_df = sel_df[['product_name', 'department', 'aisle']].head(n_items).reset_index(drop=True)
    return sel_df

def get_centroid(vector_list):
    return np.average(vector_list, axis=0)

def get_centroid_by_annoy_obj(obj_ann, obj_list):
    w2v_list = []
    for obj_id in obj_list:
        w2v_list.append(obj_ann.get_item_vector(obj_id))
    return get_centroid(w2v_list)

def get_basket_by_product_list(product_list, n_items = 15):
    selected = products[products.product_id.isin(product_list)]
    prod_vector = get_centroid(selected.vectors.tolist())
    return compose_basket_by_cart(prod_vector, input=selected.product_name.tolist(), n_items=n_items)

def get_basket_by_user_list(user_list, n_items = 15):
    user_vector = get_centroid_by_annoy_obj(u, user_list)
    return compose_basket_by_cart(user_vector, n_items=n_items)

def get_basket_by_user_product(product_list, user_list, n_items = 15):
    # Get product vector
    selected = products[products.product_id.isin(product_list)]
    prod_vector = get_centroid(selected.vectors.tolist())
    # Get 1000 nearest products
    similar_prod_1000 = p.get_nns_by_vector(prod_vector, 1000)
    sel_df = pd.DataFrame({"product_id": similar_prod_1000}).merge(products, on='product_id', how='inner')

    # Get user vector
    user_vector = get_centroid_by_annoy_obj(u, user_list)
    # Rank 1000 products by aggregated user vector
    sel_df = rank_by_euclidean(sel_df, user_vector)
    # Remove input products
    sel_df = sel_df[~sel_df.product_id.isin(selected.product_name.tolist())]
    # Return top n
    sel_df = sel_df[['product_name', 'department', 'aisle']].head(n_items).reset_index(drop=True)
    return sel_df

def get_recommendation(product_list, user_list, n_items=15):
    if product_list and user_list:
        return get_basket_by_user_product(product_list, user_list, n_items)
    elif product_list:
        return get_basket_by_product_list(product_list, n_items)
    elif user_list:
        return get_basket_by_user_list(user_list, n_items)
    else:
        return pd.DataFrame()


class RecommendationEngine:

    def __init__(self) -> None:
        pass

    def generatePredictions(self, product_list: list, user_list: list, day: str, time):
        productid_list = [PRODUCT_ID_MAP[i] for i in product_list]
        # dt = day + str(time)
        # dt_id = df_daytime_basket[df_daytime_basket.daytime ==
        #                           dt]['daytime_id'].item()
        return get_recommendation(productid_list, user_list)

    def tSNEPlot(self, selection, inputs = [],hover=None, auto_open=True, sample_size=TSNE_SIZE):

        title = 't-SNE'
        # Data sample, to speedup the execution
        df_tsne_data = products.sample(n=sample_size, random_state=42)
        df_tsne_data['size'] = 1
        df_tsne_data['color'] = 'Others'

        # Remove selection and inputs as sample
        df_tsne_data = df_tsne_data[~df_tsne_data.product_name.isin(selection)]
        df_tsne_data = df_tsne_data[~df_tsne_data.product_name.isin(inputs)]

        selection = products[products.product_name.isin(selection)].copy()  # To avoid a warning
        selection['size'] = 3
        selection['color'] = 'Recommendation'

        df_tsne_data = pd.concat([df_tsne_data, selection], ignore_index=True)

        if inputs:
            input_df = products[products.product_name.isin(inputs)].copy()
            input_df['size'] = 2
            input_df['color'] = 'Input'
            df_tsne_data = pd.concat([df_tsne_data, input_df], ignore_index=True)

        # print(df_tsne_data.head())
        # Train the TSNE MODEL
        tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=42)
        tsne_values = tsne_model.fit_transform(np.array(list(df_tsne_data['vectors'])))

        df_tsne_data['tsne-2d-one'] = tsne_values[:, 0]
        df_tsne_data['tsne-2d-two'] = tsne_values[:, 1]

        if hover is not None:
            df_tsne_data['hover'] = df_tsne_data[hover]
        else:
            df_tsne_data['hover'] = df_tsne_data[['product_name', 'aisle', 'department']].agg('<br>'.join, axis=1)

        df_tsne_data.sort_values(by='color', ascending=False, inplace=True)

        fig = px.scatter(df_tsne_data, x="tsne-2d-one", y="tsne-2d-two",
                        color='color', 
                        size="size", size_max=8,
                        title=title,
                        hover_data='hover',
                        labels={
                            # "tsne-2d-one": "Dimension one",
                            # "tsne-2d-two": "Dimension two",
                            # "color": "Type",
                            # "product_name": "Product",
                            # "department": 'Department'
                        })

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON


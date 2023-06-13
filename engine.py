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
    products, order_baskets, df_daytime_basket = load_dataframes()
    # Load annoy objects
    p, u, b, d = load_annoy_objects()
except:
    print("Unable to load files, running file generator !")

print(type(p))

PRODUCT_ID_MAP = dict(zip(products.product_name, products.product_id))

def unique_preserve_order(seq):
    """
    Return a list of unique elements from the input sequence while preserving the order of appearance.

    Args:
        seq (iterable): An iterable sequence containing elements.

    Returns:
        list: A list of unique elements from the input sequence while preserving the order of appearance.

    Examples:
        >>> unique_preserve_order([1, 2, 3, 2, 4, 3, 5, 1])
        [1, 2, 3, 4, 5]

        >>> unique_preserve_order(['apple', 'banana', 'apple', 'orange', 'banana', 'kiwi'])
        ['apple', 'banana', 'orange', 'kiwi']
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def product_lift(basket, input = None, order_baskets=order_baskets, th_support=THRESHOLD_SUPPORT, th_n=THRESHOLD_TOP, products=products):
    """
    Generate product recommendations based on the lift metric using the Apriori algorithm.

    This function takes a basket of products, along with optional input products, and calculates the lift metric to determine the strength of association between products. It then applies filters and thresholds to generate the final list of product recommendations.

    Args:
        basket (DataFrame): A DataFrame representing a basket of products.
        input (list, optional): A list of additional input products. Defaults to None.
        order_baskets (DataFrame, optional): A DataFrame representing order baskets. Defaults to the global variable `order_baskets`.
        th_support (float, optional): The minimum support threshold for the Apriori algorithm. Defaults to the global variable `THRESHOLD_SUPPORT`.
        th_n (int, optional): The maximum number of product recommendations to return. Defaults to the global variable `THRESHOLD_TOP`.
        products (DataFrame, optional): A DataFrame representing product information. Defaults to the global variable `products`.

    Returns:
        DataFrame: A DataFrame containing the product recommendations with columns 'product_name', 'department', and 'aisle'.

    """
    
    # Force to include the manual `input`
    recommendations = basket['product_id'].tolist()
    if input is not None:
        recommendations.extend(input)
    recommendations = set(recommendations)
    
    # print(recommendations)
    # Get all instances where either 1 or many products in recommendations were ordered together
    # Identify all orders where atleast 1 recommended product is available
    df_ = order_baskets[order_baskets.apply(lambda x: any(i in recommendations for i in x))].tolist()
    # For each order cart, only keep recommended products in cart
    order_baskets_ = [[i for i in sublist if i in recommendations] for sublist in df_]

    # print(order_baskets_)
    # Calculate `apriori` rules using a efficient library to speed up the calculation
    _, rules = apriori(order_baskets_, min_support=th_support, min_confidence=1e-2, max_length=5)
    
    # Multiple filters, but due to the lack of orders, are limiting the number of results, so a simple filter is active
    if input is not None:
        rules_rhs = filter(lambda rule: \
            not all(x in rule.rhs for x in input)
            , rules)
    else:
        rules_rhs = rules

    # Combine all the rules found in the data
    # Sorted by highest lift
    rule_combined = list()
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift, reverse=True):
        print(rule)
        rule_combined.extend(rule.rhs)

    # List the unique products maintaining the original order
    product_recommendation = unique_preserve_order(rule_combined)
    # print(product_recommendation)
    ## The following code, filters the recommendations after `lift`, based on the distance between the products
    # List of products
    prod = pd.DataFrame({'product_id': product_recommendation})
    prod_cross_join = prod.merge(prod, how='cross')
    # Calculate the distance between all the products
    prod_cross_join['distance'] = prod_cross_join.apply(lambda row: p.get_distance(row['product_id_x'], row['product_id_y']), axis=1)
    # Remove the same product (distance==0)
    prod_cross_join = prod_cross_join[prod_cross_join['distance']!=0]
    prod_cross_join.sort_values('distance', ascending=False)
    # Looking for closest products
    # Threshold for the filter, 10% of the distance (defined at `threshold_distance` constant)
    th_distance = np.quantile(prod_cross_join, THRESHOLD_DISTANCE)
    for id in product_recommendation:
        to_be_removed = prod_cross_join.loc[(prod_cross_join['product_id_x']==id) & (prod_cross_join['distance']<th_distance), 'product_id_y']
        prod_cross_join = prod_cross_join[~prod_cross_join['product_id_x'].isin(to_be_removed)]
    # List of final recommendations after the filters and thresholds
    prod_after_filtered = prod_cross_join['product_id_x'].unique()
    # Retain the order from the `lift`
    product_recommendation_filtered = pd.DataFrame({'product_recommendation': product_recommendation}).set_index('product_recommendation').loc[prod_after_filtered].reset_index()
    # Recall the products in the previous order
    product_recommendation_product = products.set_index("product_id").loc[product_recommendation_filtered['product_recommendation']].reset_index()

    return product_recommendation_product[['product_name', 'department', 'aisle']].head(th_n)

def basket_recompose(w2v, b=b, order_baskets=order_baskets):
    """
    Find the recommended basket based on the input Word2Vec vector.

    This function searches for a similar basket in the `b` index based on the provided Word2Vec vector. It returns a DataFrame containing the recommended basket, with each row representing an order and its associated product.

    Args:
        w2v (numpy.array): The Word2Vec vector representation of the input.
        b (AnnoyIndex, optional): An AnnoyIndex object representing the index of baskets. Defaults to the global variable `b`.
        order_baskets (DataFrame, optional): A DataFrame representing order baskets. Defaults to the global variable `order_baskets`.

    Returns:
        DataFrame: A DataFrame containing the recommended basket, with columns 'order_id' and 'product_id'.

    """
    similar_baskets = b.get_nns_by_vector(w2v, ORDER_RETURNS, search_k=-1, include_distances=False)
    basket_recompose = pd.DataFrame({'order_id': similar_baskets, 'product_id': order_baskets[similar_baskets].values}).explode('product_id')

    return basket_recompose

def get_simple_recommendation(input_vector):
    """
    Get simple product recommendations based on the input vector.

    This function takes an input vector and retrieves a list of recommended products. 
    The recommendation is obtained by finding the nearest neighbors of the input vector using the `p` index. 
    The function returns a DataFrame containing the recommended products with columns 'product_name', 'department', and 'aisle'.

    Args:
        input_vector (numpy.array): The input vector for which recommendations are generated.

    Returns:
        DataFrame: A DataFrame containing the recommended products with columns 'product_name', 'department', and 'aisle'.

    """
    product_list = p.get_nns_by_vector(input_vector, n=10)
    return products[products.product_id.isin(product_list)][['product_name', 'department', 'aisle']].reset_index(drop=True)
    
def filter_dt_recommendation(x_product = [], x_user = [], daytime_id = None):
    """
    Filter and generate recommendations based on product and user inputs.

    This function filters and generates recommendations based on product and user inputs. The function takes the following parameters:
    
    Args:
        x_product (list): A list of product IDs for which recommendations are generated. Defaults to an empty list.
        x_user (list): A list of user IDs for which recommendations are generated. Defaults to an empty list.
        daytime_id (int): The ID of the daytime for which products should be filtered. Defaults to None.

    Returns:
        tuple: A tuple containing the following elements:
            - DataFrame: The filtered and sorted product recommendations.
            - DataFrame: User basket data for the recommendations.
            - DataFrame: Product basket data for the recommendations.
            - DataFrame: The combined basket data from user and product recommendations.
            - numpy.array: The final vector representation of the recommendations.

    """
    input = None
    user_basket = None
    product_basket = None
    final_vector_list = list()
    
    basket = pd.DataFrame()
    if x_user:
        word_vector = list()
        for user in x_user:
            word_vector.append(tuple(u.get_item_vector(user)))
        user_w2v = np.average(word_vector, axis=0)
        final_vector_list.append(user_w2v)
        
        user_basket = basket_recompose(user_w2v)
        basket = pd.concat([basket, user_basket], axis=0)

    if x_product:
        word_vector = list()
        for item_id in x_product:
            word_vector.append(p.get_item_vector(item_id))
        product_w2v = np.average(word_vector, axis=0)
        final_vector_list.append(product_w2v)
        
        similar_products = p.get_nns_by_vector(product_w2v, 100 + len(x_product), search_k=-1, include_distances=False)
        product_basket = pd.DataFrame({'order_id': 0, 'product_id': similar_products})
        product_basket = product_basket[~product_basket['product_id'].isin(x_product)]
        basket = pd.concat([basket, product_basket], axis=0)
        input = x_product

    basket = basket.reset_index(drop=True).drop_duplicates('product_id')

    # If daytime is available, filter those products which were ever sold in that daytime + 4 similar daytimes
    if daytime_id is not None:
        DAYTIME_NEIGHBOURS = 10
        similar_daytime = d.get_nns_by_item(daytime_id, n=DAYTIME_NEIGHBOURS)
        filter_list_of_list = df_daytime_basket[df_daytime_basket.daytime_id.isin(similar_daytime)]['product_id'].tolist()
        filter_list = list(set([i for sublist in filter_list_of_list for i in sublist]))
        basket = basket[basket.product_id.isin(filter_list)]

    if len(final_vector_list) > 1:
        final_vector = np.average(final_vector_list, axis=0)
    else:
        final_vector = final_vector_list[0]
    
    try:
        return product_lift(basket, input), user_basket, product_basket, basket, final_vector
    except Exception as e:
        print(e)
        return get_simple_recommendation(final_vector), user_basket, product_basket, basket, final_vector


class RecommendationEngine:

    def __init__(self) -> None:
        pass

    def generatePredictions(self, product_list: list, user_list: list, day: str, time):
        productid_list = [PRODUCT_ID_MAP[i] for i in product_list]
        dt = day + str(time)
        dt_id = df_daytime_basket[df_daytime_basket.daytime ==
                                  dt]['daytime_id'].item()
        return filter_dt_recommendation(productid_list, user_list, dt_id)

    def tSNEPlot(self, selection, inputs = [],hover=None, auto_open=True, sample_size=TSNE_SIZE):

        title = 't-SNE'
        # Data sample, to speedup the execution
        df_tsne_data = products.sample(n=sample_size, random_state=42)
        df_tsne_data['size'] = 1
        df_tsne_data['color'] = 'Others'

        selection = products[products.product_name.isin(selection)].copy()  # To avoid a warning
        selection['size'] = 3
        selection['color'] = 'Recommendation'

        df_tsne_data = pd.concat([df_tsne_data, selection], ignore_index=True)
        print("INPUTS:::::", inputs)
        if inputs:
            input_df = products[products.product_name.isin(inputs)].copy()
            print(input_df)
            selection['size'] = 2
            selection['color'] = 'Input'
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


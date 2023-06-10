import plotly.express as px
from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from art import *
import numpy as np
import pandas as pd
from pandas.core.common import flatten
from annoy import AnnoyIndex
from efficient_apriori import apriori

# Load static files
order_basket = pd.read_pickle("res/order_basket.pkl")

b_annoy = AnnoyIndex(64, metric='euclidean')
b_annoy.load('res/basket_build.annoy')

d_annoy = AnnoyIndex(64, metric='euclidean')
d_annoy.load('res/daytime_build.annoy')

p_annoy = AnnoyIndex(64, metric='euclidean')
p_annoy.load('res/product_build.annoy')

u_annoy = AnnoyIndex(64, metric='euclidean')
u_annoy.load('res/user_build.annoy')

products_df = pd.read_csv("res/lemma_product.csv")
productid_map = dict(zip(products_df.product_name, products_df.product_id))

df_user_basket = pd.read_csv("res/df_user_basket.csv")
df_daytime_basket = pd.read_csv("res/df_daytime_basket.csv")

# Similarity between products
# Define the function to calculate the similarity between products

# Limiting the number of orders to process
orders_limit = 100000
# Number of orders/baskets to pull similar to the requested
orders_returns = 15
# Number of dimensions of the vector annoy is going to store.
vector_size = 64
# Number of trees for queries. When making a query the more trees the easier it is to go down the right path.
trees = 10
# Number of product recommendation as maximum
# NUMBER_OUTPUT_PRODUCTS = 10
# Sample size for the TSNE model and plot
tsne_size = 1000
# Threshold for a minimum support
threshold = 1e-3
# Threshold for the maximun number of products to bring
threshold_top = 10
# Threshold for distance, based on the quantile calculation of the basket distances
threshold_distance = 0.1

# List the unique products maintaining the original order


def unique_preserve_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# Sort recommendations by `lift`, and filter if the products are too close


def product_lift(basket, input, order_baskets=order_basket, th_support=threshold, th_n=threshold_top, products=products_df):
    # Force to include the manual `input`
    recommendations = basket['product_id'].tolist()
    if input is not None:
        recommendations.extend(input)
    recommendations = set(recommendations)

    # Baskets with only the recommended products by the w2v
    order_baskets_ = order_baskets.explode()
    order_baskets_ = order_baskets_[order_baskets_.isin(recommendations)]
    order_baskets_ = order_baskets_.groupby(level=0).apply(list)
    order_baskets_ = order_baskets_.to_list()

    # Calculate `apriori` rules using a efficient library to speed up the calculation
    _, rules = apriori(order_baskets_, min_support=th_support,
                       min_confidence=1e-2, max_length=5)

    # Multiple filters, but due to the lack of orders, are limiting the number of results, so a simple filter is active
    if input is not None:
        rules_rhs = filter(lambda rule:
                           not all(x in rule.rhs for x in input), rules)
    else:
        rules_rhs = rules

    # Combine all the rules found in the data
    # Sorted by highest lift
    rule_combined = list()
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift, reverse=True):
        # print(rule)
        rule_combined.extend(rule.rhs)

    # List the unique products maintaining the original order
    product_recommendation = unique_preserve_order(rule_combined)

    # The following code, filters the recommendations after `lift`, based on the distance between the products
    # List of products
    prod = pd.DataFrame({'product_id': product_recommendation})
    prod_cross_join = prod.merge(prod, how='cross')
    # Calculate the distance between all the products
    prod_cross_join['distance'] = prod_cross_join.apply(
        lambda row: p_annoy.get_distance(row['product_id_x'], row['product_id_y']), axis=1)
    # Remove the same product (distance==0)
    prod_cross_join = prod_cross_join[prod_cross_join['distance'] != 0]
    prod_cross_join.sort_values('distance', ascending=False)
    # Looking for closest products
    # Threshold for the filter, 10% of the distance (defined at `threshold_distance` constant)
    th_distance = np.quantile(prod_cross_join, threshold_distance)
    for id in product_recommendation:
        to_be_removed = prod_cross_join.loc[(prod_cross_join['product_id_x'] == id) & (
            prod_cross_join['distance'] < th_distance), 'product_id_y']
        prod_cross_join = prod_cross_join[~prod_cross_join['product_id_x'].isin(
            to_be_removed)]
    # List of final recommendations after the filters and thresholds
    prod_after_filtered = prod_cross_join['product_id_x'].unique()
    # Retain the order from the `lift`
    product_recommendation_filtered = pd.DataFrame({'product_recommendation': product_recommendation}).set_index(
        'product_recommendation').loc[prod_after_filtered].reset_index()
    # Recall the products in the previous order
    product_recommendation_product = products.set_index(
        "product_id").loc[product_recommendation_filtered['product_recommendation']].reset_index()

    return product_recommendation_product[['product_name', 'department', 'aisle']].head(th_n)

# Finds the recommended basket, based on the `Word2Vec` vector as input


def basket_recompose(w2v, b=b_annoy, order_baskets=order_basket):
    # Search for a similar basket in `b`
    similar_baskets = b.get_nns_by_vector(
        w2v, orders_returns, search_k=-1, include_distances=False)
    basket_recompose = pd.DataFrame(
        {'order_id': similar_baskets, 'product_id': order_baskets[similar_baskets].values}).explode('product_id')

    return basket_recompose


def basket_multi_input(product_list=[], user_list=[], daytime=None):

    print(product_list, user_list, daytime)

    basket_main = pd.DataFrame()
    input_product_list = []

    # Product list
    product_w2v = None
    if product_list:
        p_word_vector = list()
        for item_id in product_list:
            p_word_vector.append(p_annoy.get_item_vector(item_id))
        product_w2v = np.average(p_word_vector, axis=0)

        # Search for a similar basket in `b`
        basket_prod = basket_recompose(product_w2v)
        # Remove the manually selected products. Cleanup the output
        basket_prod = basket_prod[~basket_prod['product_id'].isin(
            product_list)]
        input_product_list = input_product_list + product_list
        basket_main = pd.concat([basket_main, basket_prod], axis=0)

    # User list
    selection_w2v = None
    user_list = [
        userid for userid in user_list if userid in df_user_basket.user_id.tolist()]
    if user_list:
        u_word_vector = list()
        for item_id in user_list:
            u_word_vector.append(tuple(u_annoy.get_item_vector(item_id)))
        user_w2v = np.average(u_word_vector, axis=0)
        basket = basket_recompose(user_w2v)
        # Products from the list of users
        input = df_user_basket.loc[df_user_basket['user_id'].isin(
            user_list), 'product_id']
        input = [item for sublist in input for item in sublist]
        input_product_list = input_product_list + input

        basket_main = pd.concat([basket_main, basket], axis=0)

    if daytime is not None:
        daytime_w2v = d_annoy.get_item_vector(daytime)
        basket_dt = basket_recompose(daytime_w2v)

        input = df_daytime_basket.loc[df_daytime_basket['daytime_id']
                                      == daytime, 'product_id'].item()
        input_product_list = input_product_list + [input]
        basket_main = pd.concat([basket_main, basket_dt], axis=0)

    return product_lift(basket_main, input_product_list)


class RecommendationEngine:

    def __init__(self) -> None:
        pass

    def generatePredictions(self, product_list: list, user_list: list, day: str, time):
        productid_list = [productid_map[i] for i in product_list]
        dt = day + str(time)
        dt_id = df_daytime_basket[df_daytime_basket.order_daytime ==
                                  dt]['daytime_id'].item()
        return basket_multi_input(productid_list, user_list, dt_id)

    def getTSNEPlot():
        pass

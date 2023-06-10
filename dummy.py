import pandas as pd
import numpy as np

import json
import plotly
import plotly.express as px
from sklearn.manifold import TSNE

data = {
    'Product': [
        'Organic Broccoli',
        'Organic Extra Firm Tofu',
        'Organic Hass Avocado',
        'Sonoma Traditional Flour Tortillas 10 Count',
        'Carrots',
        'Organic Baby Carrots',
        'Banana',
        'Organic Beef Broth',
        'Organic Coconut Milk',
        'Pineapple Chunks'
    ],
    'Department': [
        'produce',
        'deli',
        'produce',
        'bakery',
        'produce',
        'produce',
        'produce',
        'canned goods',
        'dairy eggs',
        'frozen'
    ],
    'Aisle': [
        'fresh vegetables',
        'tofu meat alternatives',
        'fresh fruits',
        'tortillas flat bread',
        'fresh vegetables',
        'packaged vegetables fruits',
        'fresh fruits',
        'soup broth bouillon',
        'soy lactosefree',
        'frozen produce'
    ]
}

DUMMY_TABLE_DF = pd.DataFrame(data)

# PRODUCTS_DF = pd.read_csv("lemma_product.csv")


def bar_with_plotly():

   # Students data available in a list of list
    students = [['Akash', 34, 'Sydney', 'Australia'],
                ['Rithika', 30, 'Coimbatore', 'India'],
                ['Priya', 31, 'Coimbatore', 'India'],
                ['Sandy', 32, 'Tokyo', 'Japan'],
                ['Praneeth', 16, 'New York', 'US'],
                ['Praveen', 17, 'Toronto', 'Canada']]

    # Convert list to dataframe and assign column values
    df = pd.DataFrame(students,
                      columns=['Name', 'Age', 'City', 'Country'],
                      index=['a', 'b', 'c', 'd', 'e', 'f'])

    # Create Bar chart
    fig = px.bar(df, x='Name', y='Age', color='City', barmode='group')

    # Create graphJSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def plotly_tsne(dummy_data=None, labels=None):
    if dummy_data is None:
        # Generate dummy data
        dummy_data = np.random.randn(100, 10)

        # Create random labels
        labels = np.random.choice(['A', 'B', 'C'], size=100)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(dummy_data)

    # Create a DataFrame with the t-SNE results
    tsne_df = pd.DataFrame(
        {'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'label': labels})

    # Create the plot using Plotly Express
    fig = px.scatter(tsne_df, x='x', y='y', color='label')
    # Create graphJSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

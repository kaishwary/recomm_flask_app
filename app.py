"""
app.py

This file contains the main Flask application for the recommendation engine web service. It defines the routes, form classes, and functions necessary for handling user requests and generating recommendations.

The code performs the following tasks:
- Imports necessary modules and libraries for the Flask application.
- Defines a Flask application object and sets the secret key.
- Initializes the Bootstrap extension for Flask.
- Creates an instance of the RecommendationEngine class.
- Defines a FlaskForm class for collecting product information and user details.
- Defines a function to fetch options from the product list based on a search query.
- Defines route handlers for serving HTML templates, loading search options dynamically via AJAX call, and processing form submissions.

Utility Functions:
- fetch_options: Fetches options from the product list based on a search query.

Routes:
- '/load_options': Route to dynamically serve search options via an AJAX call.
- '/': Main route for the web service. Handles GET and POST requests for the recommendation form.

File name: app.py
"""

from engine import RecommendationEngine, products
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap

from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SelectMultipleField


app = Flask(__name__)
app.config['SECRET_KEY'] = 'SECRET_KEY'  # TODO - Later
bootstrap = Bootstrap(app)
# recommendation engine
r_engine = RecommendationEngine()
# product list - to be used for populating search box
pl = products.product_name.to_list()


class ProductForm(FlaskForm):
    """
    Represents a form for collecting product information and user details.

    This form includes fields for selecting multiple products, entering a user ID, choosing a day, and selecting a time.

    Attributes:
        product (SelectMultipleField): A multiple-choice field for selecting products.
        user_id (StringField): A text input field for entering the user ID.
        day (SelectField): A dropdown menu for selecting a day of the week.
        time (SelectField): A dropdown menu for selecting a time of the day.
    """
    product = SelectMultipleField('Enter Product', default=None)
    user_id = StringField('Enter User ID', default=None)
    day = SelectField('Day', choices=[('Monday', 'Monday'), ('Tuesday', 'Tuesday'), ('Wednesday', 'Wednesday'), (
        'Thursday', 'Thursday'), ('Friday', 'Friday'), ('Saturday', 'Saturday'), ('Sunday', 'Sunday')])
    time = SelectField('Time', choices=[
                       (f'{hour:02d}00', f'{hour:02d}:00') for hour in range(24)], coerce=int)


def fetch_options(search_query=None):
    """
    Fetches options from product list based on a search query.
    Called when search term is entered in Product Select box

    Args:
        search_query (str): The search query to filter the options. Defaults to None.

    Returns:
        list: A list of dictionaries representing the options. Each dictionary has "value" and "label" keys.

    """
    # print(search_query)
    if search_query is None or search_query == '':
        return [{"id": option, "text": option} for option in pl[:10]]
    else:
        # print([option for option in pl if search_query.lower() in option.lower()][:5])
        return [{"id": option, "text": option} for option in pl if search_query.lower() in option.lower()][:50]
        # return [option for option in pl if search_query.lower() in option.lower()][:50]

# Route to dynamically serve search options via AJAX call
@app.route('/load_options', methods=['GET'])
def load_options():
    search_query = request.args.get('term')
    # Fetch and filter options based on the search query
    options = fetch_options(search_query)
    return jsonify({'results': options})

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    form = ProductForm()

    if request.method == 'POST':
        # Process the form data
        # Product list
        product_input = form.product.data
        # user ID list
        user_id_raw = form.user_id.data
        if user_id_raw == '':
            user_id = []
        else:
            user_id = [int(i.strip()) for i in user_id_raw.split(',')]

        day = form.day.data
        time = int(form.time.data / 100)

        # Generate input datafarme
        input_df = pd.DataFrame.from_dict({
            "Product(s)": ", ".join(product_input),
            "User ID(s)": user_id_raw,
            "Day of Week": day,
            "Time of Day (hrs)": form.time.data
        }, orient='index', columns=['Input']).to_html(classes='table table-striped table-hover',
                                                      index=True, justify='center')

        # Generate predictions
        df, _, _, _, _ = r_engine.generatePredictions(product_input, user_id, day, time)
        htmlTable = df.to_html(
            classes='table table-striped table-hover',
            index=False, justify='center')

        graphJSON = r_engine.tSNEPlot(selection = df.product_name.tolist(), inputs=product_input)
        # Generate tSNE plot
        # graphJSON = plotly_tsne()

        errors = []
        for field_name, field_errors in form.errors.items():
            errors.extend(field_errors)

        # handle the specific validation errors
        # based on the field names or display a generic error message
        print("Validation errors:", errors)

        if not errors:
            return render_template('result.html', resulttable=[htmlTable], titles=['na'],
                                   inputtable=[input_df],
                                   graphJSON=graphJSON)
        # TODO - add error handling

    return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)

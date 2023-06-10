from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from flask_material import Material
from forms import ProductForm
from dummy import plotly_tsne

import pandas as pd
from engine import RecommendationEngine, products_df


app = Flask(__name__)
app.config['SECRET_KEY'] = 'SECRET_KEY'  # TODO - Later
bootstrap = Bootstrap(app)
# recommendation engine
r_engine = RecommendationEngine()
# product list - to be used for populating search box
pl = products_df.product_name.to_list()


def fetch_options(search_query=None):
    # print(search_query)
    if search_query is None or search_query == '':
        return [{"value": option, "label": option} for option in pl[:10]]
    else:
        # print([option for option in pl if search_query.lower() in option.lower()][:5])
        return [{"value": option, "label": option} for option in pl if search_query.lower() in option.lower()][:50]
        # return [option for option in pl if search_query.lower() in option.lower()][:50]

# TODO - remove once testing done


@app.route('/base', methods=['GET'])
def serve_base():
    return render_template('base.html')

# TODO - remove once testing done


@app.route('/index', methods=['GET'])
def serve_index():
    return render_template('index.html')

# Route to dynamically serve search options via AJAX call


@app.route('/load_options', methods=['GET'])
def load_options():
    search_query = request.args.get('searchQuery')
    # Fetch and filter options based on the search query
    options = fetch_options(search_query)
    return jsonify({'options': options})

# Main route


@app.route('/', methods=['GET', 'POST'])
def index():
    form = ProductForm()

    if request.method == 'POST':
        # Process the form data
        # Product list
        product = form.product.data
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
            "Product(s)": ", ".join(product),
            "User ID(s)": user_id_raw,
            "Day of Week": day,
            "Time of Day (hrs)": form.time.data
        }, orient='index', columns=['Input']).to_html(classes='table table-striped table-hover',
                                                      index=True, justify='center')

        # Generate predictions
        df = r_engine.generatePredictions(product, user_id, day, time)
        htmlTable = df.to_html(
            classes='table table-striped table-hover',
            index=False, justify='center')

        # Generate tSNE plot
        graphJSON = plotly_tsne()

        errors = []
        for field_name, field_errors in form.errors.items():
            errors.extend(field_errors)

        # You can now handle the specific validation errors
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

from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SelectField, RadioField, SelectMultipleField
from wtforms.validators import InputRequired


class ProductForm(FlaskForm):
    product = SelectMultipleField('Enter Product', default=None)
    user_id = StringField('Enter User ID', default=None)
    # use_current_datetime = RadioField('Use current day-time', choices=[('yes', 'Yes'), ('no', 'No')], default='yes')
    day = SelectField('Day', choices=[('Monday', 'Monday'), ('Tuesday', 'Tuesday'), ('Wednesday', 'Wednesday'), (
        'Thursday', 'Thursday'), ('Friday', 'Friday'), ('Saturday', 'Saturday'), ('Sunday', 'Sunday')])
    time = SelectField('Time', choices=[
                       (f'{hour:02d}00', f'{hour:02d}:00') for hour in range(24)], coerce=int)

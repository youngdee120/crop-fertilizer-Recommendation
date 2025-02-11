from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from wtforms import FloatField
import pandas as pd
import pickle
import numpy as np
import sklearn
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For session and flash messages
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Load crop model
crop_model = joblib.load('models/crop_model.pkl')
sc = pickle.load(open('models/standscaler.pkl','rb'))
ms = pickle.load(open('models/minmaxscaler.pkl','rb'))

# Load new fertilizer model
fertilizer_model = pickle.load(open('models/classifier.pkl','rb'))
ferti = pickle.load(open('models/fertilizer.pkl','rb'))

# Database model for users
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Initialize the database
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template("login.html")

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email')
    password = request.form.get('password')

    # Check if user already exists
    user = User.query.filter_by(email=email).first()
    if user:
        flash('Email already registered. Please login.')
        return redirect(url_for('home'))

    # Hash password and save user
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    flash('Account created successfully! Please log in.')
    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    # Verify user credentials
    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password, password):
        session['user_id'] = user.id  # Store user session
        return redirect(url_for('landing'))
    else:
        flash('Invalid email or password. Please try again.')
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user session
    flash("You have been logged out.")
    return redirect(url_for('home'))  # Redirect to login page

@app.route('/landing')
def landing():
    if 'user_id' not in session:
        flash("Please log in first.")
        return redirect(url_for('home'))  # Restrict access if not logged in
    return render_template("landing.html")

@app.route('/Detail')
def Detail():
    return render_template('Detail.html')

@app.route('/Model1')
def Model1():
    return render_template('Model1.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    mois = request.form.get('mois')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')
    
    if None in (temp, humi, mois, soil, crop, nitro, pota, phosp) or not all(val.isdigit() for val in (temp, humi, mois, soil, crop, nitro, pota, phosp)):
        return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')

    # Convert values to integers
    input = [int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]
    res = ferti.classes_[fertilizer_model.predict([input])]
    return render_template('Model1.html', x=res)

class CropForm(FlaskForm):
    Nitrogen = FloatField('Nitrogen')
    Phosphorus = FloatField('Phosphorus')
    Potassium = FloatField('Potassium')
    Temperature = FloatField('Temperature')
    Humidity = FloatField('Humidity')
    Ph = FloatField('pH')
    Rainfall = FloatField('Rainfall')

@app.route("/crops", methods=['GET', 'POST'])
def crops():
    if 'user_id' not in session:
        flash("Please log in first.")
        return redirect(url_for('home'))

    form = CropForm()
    if form.validate_on_submit():
        try:
            # Retrieve form data
            N = form.Nitrogen.data
            P = form.Phosphorus.data
            K = form.Potassium.data
            temp = form.Temperature.data
            humidity = form.Humidity.data
            ph = form.Ph.data
            rainfall = form.Rainfall.data

            # Prepare feature array
            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            # Scale features
            scaled_features = ms.transform(single_pred)
            final_features = sc.transform(scaled_features)

            # Predict crop
            prediction = crop_model.predict(final_features)

            # Map prediction to crop name
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
                6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
                11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
                15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            crop = crop_dict.get(prediction[0], "Unknown")
            result = f"{crop} is the best crop to be cultivated right there."

        except Exception as e:
            result = f"Error: {str(e)}"
            flash(result, 'error')

        return render_template('crops.html', form=form, result=result)

    return render_template('crops.html', form=form, result=None)

# Prevent back button after logout
@app.after_request
def add_no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template
from project.web_app.backend import db

app = Flask(__name__, template_folder='views')

#define the home page of the web application
@app.route('/')
def home():
    num_of_restaurants = db.find_count('restaurants')
    num_of_users = db.find_count('users')
    num_of_reviews = db.find_count('reviews')

    return render_template('home.html', num_of_restaurants=num_of_restaurants, num_of_users=num_of_users,
                           num_of_reviews=num_of_reviews)


#define the home page of the web application
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
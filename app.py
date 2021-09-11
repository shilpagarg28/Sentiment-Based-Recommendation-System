import flask
from flask import Flask, jsonify,  request, render_template
from model import getItemsForUser

# Create the application.
app = flask.Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user_name =  request.form["user_name"]
        print(user_name)
        output = getItemsForUser(user_name)
        return render_template('index.html', prediction_text='Suggested Items {}'.format(output))
    else :
        return render_template('index.html')    

if __name__ == '__main__':
    app.debug=False
    app.run()

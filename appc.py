from flask import Flask, request, render_template
from flask_mail import Mail, Message
import pickle
import sklearn
import pandas as pd
import random
import numpy as np

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'xyz@gmail.com'
app.config['MAIL_PASSWORD'] = 'xyz'
app.config['MAIL_USE_SSL'] = True
#connect mail class to our app
mail = Mail(app)

@app.route("/")
def home():
    return render_template("index.html")
#Get request
#Post request
@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
		print('Got Data from Client: ', dict(request.get_json(force = True)))

		data = dict(request.get_json(force = True))

		#Loading model
		with open('CreditCard.pkl', 'rb') as file:
			pickle_model = pickle.load(file)

		#Loading csv file for finding the match with the input data
		xTest = pd.read_csv('creditcard.csv')

		try:
			time_data = float(data['time'].strip())
			amount_data = float(data['amount'].strip())
		except ValueError as e:
			return render_template('index.html', pre = 'Invalid Input')

		#Get the matching row
		pca_credit = xTest[(xTest['time'] == float(data['time'])) & (xtest['amount'] == float(data['amount']))]

		if (len(pca_credit) == 0):
			return render_template('index.html', pre = 'Invalid Data')

		required = np.array(pca_credit)
		testData = required[0][:-1].reshape(1, -1)

		pickle_model.decision_function(testData)
		output = pickle_model.predict(testData)
		
		#Creating a constructor for message
		msg = Message(
			'Transaction Report',
			sender = 'xyz@gmail.com',
			recipients = ['abc@gmail.com'])

		# To send a message
		if (required[0][-1] == 1.0 and output == [-1]):
			msg.body = 'It seems to be a fradulent transaction'
			mail.send(msg)
			return render_template('index.html', pre = 'It seems to be a fradulent transaction')
		elif(required[0][-1] == 0.0 and output == [1]):
			msg.body = 'It is a normal transaction'
			mail.send(msg)
			return render_template('index.html', pre = 'It is a normal transaction')
		else:
			msg.body = 'It seems to be fraudulent transaction'
			mail.send(msg)
			return render('index.html', pre = 'It seems to be fraudulent transaction')
	return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
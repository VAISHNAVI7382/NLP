# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:29:19 2019

@author: lalit
"""
    
from flask import render_template, Flask, request,url_for
from keras.models import load_model
import pickle 
import tensorflow as tf
graph = tf.get_default_graph()
with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cla = load_model('zomato.h5',compile=False)
#cla.compile(optimizer='adam',loss='categorical_crossentropy')
app = Flask(__name__)
@app.route('/')
def index():
	return render_template('index2.html')

@app.route('/tpredict',methods = ["POST"])
def page2():
		if request.method =="POST":

			topic = request.form['review']
			topic=cv.transform([topic])
			nameslist=["Average","Excellent" , "Fair","Good","Poor"]
			with graph.as_default():
				y_pred = cla.predict_classes(topic)
		
		return render_template('index2.html',y = str(nameslist[y_pred[0]]))
        
    	   
	



if __name__ == '__main__':
    app.run(host = 'localhost', debug = True , threaded = False)
    

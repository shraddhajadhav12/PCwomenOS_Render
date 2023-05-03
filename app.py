from flask import Flask,render_template,redirect, url_for,request
import numpy as np
import pickle
import tensorflow as tf
import cv2
import gunicorn
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/text-test.html')
def text_test():
    return render_template('text-test.html')

@app.route('/image-test.html')
def image_test():
    return render_template('image-test.html')

@app.route('/read_more.html')
def about_us():
    return render_template('read_more.html')


 
# A decorator used to tell the application
# which URL is associated function
@app.route('/text_prediction', methods =["GET", "POST"])
def gfg():
    list1=[]
    if request.method == "POST":
       regular_periods = request.form.get("rp")
       regular_periods=int(regular_periods)
       list1.append(regular_periods)

    #    painful_periods = request.form.get("pp")
    #    painful_periods=int(painful_periods)
    #    list1.append(painful_periods)

       period_cycle=request.form.get("pc")
       period_cycle=int(period_cycle)
       list1.append(period_cycle)

       period_pain_scale=request.form.get("pps")
       period_pain_scale=int(period_pain_scale)
       list1.append(period_pain_scale)

       hair_growth=request.form.get("ehg")
       hair_growth=int(hair_growth)
       list1.append(hair_growth)

       stress=request.form.get("s")
       stress=int(stress)
       list1.append(stress)

       feature=np.array([list1])
       model = pickle.load(open('KNN_retraining_model.pkl','rb'))
       model.predict(feature)
       if(model.predict(feature)==0):
           predict_val="NON-PCOS"
       else:
           predict_val="PCOS"
    if(predict_val=="PCOS"):
        return render_template("pcos.html")
    else:
        return render_template("non-pcos.html")
    
@app.route('/prediction', methods =["POST"])
def prediction():
       IMAGE_SIZE=(224,224)
       user_img=request.files['img']
       user_img.save("test_img.jpg")
       test_img1 = cv2.imread("test_img.jpg")
       ultrasound_model=tf.keras.models.load_model(r"Final_Model")
       test_img1 = cv2.resize(test_img1, IMAGE_SIZE)
       test_img1 = test_img1[np.newaxis, :]
       prediction = ultrasound_model.predict(test_img1)
       if np.argmax(prediction) == 0:
           return render_template("pcos.html")
       else:
           return render_template("non-pcos.html")

if __name__=="__main__":
    app.run(debug=True,port=5000)

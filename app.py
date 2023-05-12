from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os
from keras.models import load_model
from keras.utils import load_img
import cv2
from PIL import Image
import numpy as np
from numpy import array

app = Flask(__name__)

age_model=load_model('age-12-6.26.hdf5')
gender_model=load_model('gender-43-0.23.hdf5')
race_model=load_model('race-40-0.58.hdf5')

race_list=['White', 'Black', 'Asian', 'Indian', 'Others']
gender_list=['Male', 'Female']

mnist_model=load_model("mnist.h5")

cifar_10_model=load_model("cifar_10.h5")
cifar_list=['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///records.db'
db=SQLAlchemy(app)
class faces(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    picture=db.Column(db.String)
    #age=db.Column(db.Integer)
    #gender=db.Column(db.String)
    #race=db.Column(db.String)
    def __init__(self, picture):
        self.picture=picture
        #self.age=age
        #self.gender=gender
        #self.race=race
class mnists(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    picture=db.Column(db.String)
    mnist_number=db.Column(db.Integer)
    def __init__(self, picture, mnist_number):
        self.picture=picture
        self.mnist_number=mnist_number
class cifars(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    picture=db.Column(db.String)
    cifar_category=db.Column(db.String)
    def __init__(self, picture, cifar_category):
        self.picture=picture
        self.cifar_category=cifar_category

with app.app_context():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")
#UTK
@app.route('/face_page', methods=['GET', 'Post'])
def face_page():
    return render_template("face.html", faces = faces.query.all())

@app.route('/add_face', methods=['GET', 'POST'])
def add_face():
    if request.method=='POST': 
        img=request.files['image_face']
        img_path="static/UTK/" + img.filename
        img.save(img_path)
        img_path=img_path.replace('"', '')
        face=faces(picture=img_path)
        db.session.add(face)
        db.session.commit()
        return redirect(url_for("face_page"))            
    return render_template("add_face.html")

@app.route('/delete_face/<int:id>', methods=['GET', 'POST'])
def delete_face(id):
    delete_face=faces.query.get_or_404(id)
    db.session.delete(delete_face)
    db.session.commit()
    os.remove(delete_face.picture)
    return redirect(url_for("face_page"))

def face_predict(img_path):
    X=cv2.imread(img_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    result=[]
    i=0
    for (x,y,w,h) in faces:
        pred_result={}
        i = i+1
        cv2.rectangle(X,(x,y),(x+w,y+h),(255,0,0),2)
        X_face=gray[y:y+h,x:x+w]
        
        X_resized=cv2.resize(X_face, (32, 32), interpolation = cv2.INTER_AREA)
        X_flatten=array(X_resized).flatten()
        X_flatten_reshaped = X_flatten.reshape(32,32,1).astype('float32') 
        X_flatten_reshaped_normalized=X_flatten_reshaped/255
        
        age_pred_prob=age_model.predict(X_flatten_reshaped_normalized.reshape(1,32,32,1))
        age=str(np.round(age_pred_prob)).replace('[','').replace(']','').replace('.','')
        pred_result["Age: "]=age

        gender_pred_prob=gender_model.predict(X_flatten_reshaped_normalized.reshape(1,32,32,1))
        gender=str(np.round(gender_pred_prob)).replace('[','').replace(']','').replace('.','')
        gender=gender_list[int(gender)]
        if gender=='Male':
            gender_prob="%.2f"%(100-gender_pred_prob*100)
        else:
            gender_prob="%.2f"%(gender_pred_prob*100)
        pred_result["Gender: "]=gender+'('+gender_prob+'%)'

        race_pred_prob=race_model.predict(X_flatten_reshaped_normalized.reshape(1,32,32,1))
        race=str(np.argmax(race_pred_prob, axis=1)).replace('[','').replace(']','')
        race=race_list[int(race)]
        race_prob="%.2f"%(np.max(race_pred_prob)*100)
        pred_result["Race: "]=race+'('+race_prob+'%)'

        color = (0,255,0)
        cv2.putText(X, str(i), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 4, color, 4)
        
        face_color = X[y:y+h, x:x+w]
        eyes=eye_cascade.detectMultiScale(X_face, 1.1,3)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(face_color, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        result.append(pred_result)
    new_img=Image.fromarray(cv2.cvtColor(X, cv2.COLOR_BGR2RGB))
    new_img.save('static/UTK/temp.png')
    return result

@app.route('/predict_face/<int:id>', methods=['GET', 'POST'])
def predict_face(id):
    face=faces.query.filter_by(id=id).first()
    img_path=face.picture	
    prediction=face_predict(img_path)
    img='UTK/temp.png'
    return render_template("predict_face.html", prediction=prediction, img=img, face=face)

#MNIST
@app.route('/mnist_page', methods=['GET', 'Post'])
def mnist_page():
    return render_template("mnist.html", mnists = mnists.query.all())

@app.route('/draw_mnist', methods=['GET', 'Post'])
def draw_mnist():
    return render_template("draw_mnist.html")

@app.route('/add_mnist', methods=['GET', 'POST'])
def add_mnist():
    if request.method=='POST': 
        img=request.files['image_mnist']
        img_path="static/MNIST/" + img.filename
        img.save(img_path)
        img_path=img_path.replace('"', '')
        mnist=mnists(picture=img_path , mnist_number=request.form['mnist_number'])
        db.session.add(mnist)
        db.session.commit()
        return redirect(url_for("mnist_page"))     
    return render_template("add_mnist.html")

@app.route('/delete_mnist/<int:id>', methods=['GET', 'POST'])
def delete_mnist(id):
    delete_mnist=mnists.query.get_or_404(id)
    db.session.delete(delete_mnist)
    db.session.commit()
    os.remove(delete_mnist.picture)
    return redirect(url_for("mnist_page"))

def mnist_predict(img_path):
    img=load_img(img_path, color_mode="grayscale")
    img=img.resize((28, 28),resample=Image.LANCZOS)
    img_flatten=array(img).flatten()
    img_flatten_reshaped = img_flatten.reshape(1,28,28,1).astype('float32') 
    img_flatten_reshaped_normalized=img_flatten_reshaped/255
    pred_prob=mnist_model.predict(img_flatten_reshaped_normalized)
    mnist_num=str(np.argmax(pred_prob,axis=1)).replace('[','').replace(']','')
    mnist_num_prob="%.2f"%(np.max(pred_prob)*100)
    result=mnist_num+" ("+mnist_num_prob+"%)"
    return result

@app.route('/predict_mnist/<int:id>', methods=['GET', 'POST'])
def predict_mnist(id):
    mnist=mnists.query.filter_by(id=id).first()
    img_path=mnist.picture	
    prediction=mnist_predict(img_path)
    img_path=img_path.replace('static/','')
    return render_template("predict_mnist.html", prediction=prediction, img_path=img_path, mnist=mnist)

#CIFAR-10
@app.route('/cifar_10_page', methods=['GET', 'Post'])
def cifar_10_page():
    return render_template("cifar_10.html", cifars = cifars.query.all())

@app.route('/add_cifar', methods=['GET', 'POST'])
def add_cifar():
    if request.method=='POST': 
        img=request.files['image_cifar_10']
        img_path="static/CIFAR-10/" + img.filename
        img.save(img_path)
        img_path=img_path.replace('"', '')
        cifar=cifars(picture=img_path , cifar_category=request.form['cifar_10_category'])
        db.session.add(cifar)
        db.session.commit()
        return redirect(url_for("cifar_10_page"))            
    return render_template("add_cifar.html")

@app.route('/delete_cifar/<int:id>', methods=['GET', 'POST'])
def delete_cifar(id):
    delete_cifar=cifars.query.get_or_404(id)
    db.session.delete(delete_cifar)
    db.session.commit()
    os.remove(delete_cifar.picture)
    return redirect(url_for("cifar_10_page"))

def cifar_predict(img_path):
    img=Image.open(img_path)
    img=img.resize((32, 32),resample=Image.LANCZOS)
    img_flatten=array(img).flatten()
    img_flatten_reshaped = img_flatten.reshape(1,32,32,3).astype('float32') 
    img_flatten_reshaped_normalized=img_flatten_reshaped/255
    pred_prob=cifar_10_model.predict(img_flatten_reshaped_normalized)
    cifar_label=str(np.argmax(pred_prob,axis=1)).replace('[','').replace(']','')
    cifar_label=cifar_list[int(cifar_label)]
    cifar_prob="%.2f"%(np.max(pred_prob)*100)
    result=cifar_label+" ("+cifar_prob+"%)"
    return result

@app.route('/predict_cifar/<int:id>', methods=['GET', 'POST'])
def predict_cifar(id):
    cifar=cifars.query.filter_by(id=id).first()
    img_path=cifar.picture	
    prediction=cifar_predict(img_path)
    img_path=img_path.replace('static/','')
    return render_template("predict_cifar.html", prediction=prediction, img_path=img_path, cifar=cifar)

if __name__ == '__main__':
    app.debug = True
    app.run()
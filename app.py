import time

import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from playsound import playsound

import cvlib as cv
from cvlib.object_detection import draw_bbox

app = Flask(__name__)

# Create a list to store registered users
users = []


# Default home page or route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index.html')
def home():
    return render_template("index.html")


# Registration page
@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/afterreg', methods=['POST'])
def afterreg():
    name = request.form.get('name')
    user_id = request.form.get('user_id')
    password = request.form.get('password')

    # Check if the user is already registered
    for user in users:
        if user['user_id'] == user_id:
            return render_template('register.html', pred="You are already a member, please login using your details")

    # Create a new user dictionary
    user = {
        'name': name,
        'user_id': user_id,
        'password': password
    }

    # Append the new user to the users list
    users.append(user)

    return render_template('register.html', pred="Registration Successful, please login using your details")


# Login page
@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/afterlogin', methods=['POST'])
def afterlogin():
    user_id = request.form.get('user_id')
    password = request.form.get('password')

    # Find the user in the users list
    for user in users:
        if user['user_id'] == user_id and user['password'] == password:
            return redirect(url_for('prediction'))

    return render_template('login.html', pred="Invalid username or password")


@app.route('/logout')
def logout():
    return render_template('logout.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/result', methods=["GET", "POST"])
def res():
    webcam = cv2.VideoCapture('drowning.mp4')

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    t0 = time.time()  # gives time in seconds after 1970

    # variable dcount stands for how many seconds the person has been standing still for
    centre0 = np.zeros(2)
    isDrowning = False

    # This loop happens approximately every 1 second, so if a person doesn't move,
    # or moves very little for 10 seconds, we can say they are drowning

    # Loop through frames
    while webcam.isOpened():
        # Read frame from webcam
        status, frame = webcam.read()

        if not status:
            print("Could not read frame")
            exit()

        # Apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)

        if len(bbox) > 0:
            bbox0 = bbox[0]
            centre = [(bbox0[0] + bbox0[2]) / 2, (bbox0[1] + bbox0[3]) / 2]

            # Make vertical and horizontal movement variables
            hmov = abs(centre[0] - centre0[0])
            vmov = abs(centre[1] - centre0[1])

            x = time.time()

            threshold = 10
            if hmov > threshold or vmov > threshold:
                print(x - t0, 's')
                t0 = time.time()
                isDrowning = False
            else:
                print(x - t0, 's')
                if time.time() - t0 > 10:
                    isDrowning = True

            centre0 = centre

        out = draw_bbox(frame, bbox, label, conf, isDrowning)

        cv2.imshow("Real-time object detection", out)

        if isDrowning:
            playsound('alarm.mp3')
            webcam.release()
            cv2.destroyAllWindows()
            return render_template('prediction.html', prediction="Emergency !!! The Person is drowning")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    webcam.release()
    cv2.destroyAllWindows()


import subprocess
@app.route('/predict', methods=["GET", "POST"])
def predict():
    subprocess.call(['python', '/Users/yatharthsingh/Desktop/Project/Final Deliverables/detect.py', '--weights', '/Users/yatharthsingh/Desktop/Project/Final Deliverables/best.pt', '--img', '640', '--conf', '0.25', '--source', '/Users/yatharthsingh/Desktop/Project/Final Deliverables/drowning.mp4'])
    return render_template('test.html')
if __name__ == "__main__":
    app.run(debug=True)
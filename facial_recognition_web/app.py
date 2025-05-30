# app.py
from flask import Flask, render_template, Response, request, redirect, url_for, flash
import threading
from recognizer import register_face, gen_frames, reload_faces_after_registration

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for flashing messages

# Flag and threading to handle registration without blocking video stream
registering = False
register_name = None

def registration_thread(name):
    global registering
    register_face(name)
    reload_faces_after_registration()  # Reload embeddings after registration
    registering = False

@app.route('/', methods=['GET', 'POST'])
def index():
    global registering, register_name
    if request.method == 'POST':
        name = request.form.get('name')
        if not name:
            flash("Please enter a valid name!", "error")
        else:
            if not registering:
                registering = True
                register_name = name
                thread = threading.Thread(target=registration_thread, args=(name,))
                thread.start()
                flash(f"Started face registration for {name}. Please look at the camera and press 'c' to capture images.", "info")
            else:
                flash("Already registering a user. Please wait.", "warning")
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(lambda: registering),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

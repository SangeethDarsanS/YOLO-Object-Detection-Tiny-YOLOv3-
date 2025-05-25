# main.py
from flask import Flask, render_template, request, redirect
from detect import run_detection
import threading

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start', methods=['POST'])
def start():
    thread = threading.Thread(target=run_detection)
    thread.start()
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)

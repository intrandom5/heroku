from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        user = 'rando'
        data = {'model' : 'deepspeech', 'wav' : '0001.wav', 'script' : 'result.txt'}
        data['wav'] = request.form.get('wav')
        return render_template('index.html', user=user, data=data)
    elif request.method == "GET":
        user = 'rando'
        data = {'model': 'deepspeech', 'wav': '0001.wav', 'script': 'result.txt'}
        return render_template('index.html', user=user, data=data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
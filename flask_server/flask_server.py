from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return ('this is working bitch!')
@app.route('/get')
def predict():
    ## TODO: receive operation mode and the picture to predict

if __name__ == "__main__":
    app.run()

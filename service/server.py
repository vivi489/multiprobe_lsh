from flask import Flask, render_template, request, url_for
from flask import jsonify
from backend import *

app = Flask(
    __name__
)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/send', methods = ['POST'])
def send():
    query = request.form["search-query"]
    candidates = lookup_knn(
        query,
        top_k=20
    )
    return render_template('answers.html', candidates=enumerate(candidates))

@app.route('/_pickone', methods= ['GET'])
def sample_query():
    print("_pickone requested!")
    return jsonify(pick_one_query())

def main():
    app.run(host="0.0.0.0", debug=True)


if __name__ == "__main__":
    main()

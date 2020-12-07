from flask import Flask, render_template, request, url_for
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
    candidates = lookup(
        query,
        top_k=20
    )
    return render_template('answers.html', candidates=enumerate(candidates))

def main():
    app.run(host="0.0.0.0", debug=True)


if __name__ == "__main__":
    main()

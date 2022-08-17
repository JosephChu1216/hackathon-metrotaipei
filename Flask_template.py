from flask import Flask, render_template
import json

app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route("/", methods=['GET', 'POST'])
@app.route("/index.html", methods=['GET', 'POST'])
def index():
    username = 'Joseph'
    with open('data.json') as jsonfile:
        dataInfo = json.load(jsonfile)
    return render_template('index.html', username=username, dataInfo=dataInfo)


# @app.route("/<name>")
# def homename(name):
#     return f"<h1>hello {name}</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
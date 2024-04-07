from flask import Flask, render_template

app = Flask(__name__, static_url_path='/static')


@app.route("/")
def hello_world():
    return render_template("home.html")

@app.route("/home")
def hello_dush(): 
    return render_template("dush.html")

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect
from flask.helpers import url_for
from toneAnalyze import toneAnalyze

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def tone():
    if request.args.get('sentence') == None:
        predicted_tone = None
    else:
        predicted_tone = toneAnalyze(
            request.args.get('sentence'))

    return render_template("index.html", tone_pred=predicted_tone)


if __name__ == "__main__":
    app.run(debug=True)

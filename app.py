from flask import Flask, request, render_template
from flask_cors import CORS
from hybrid_matcher import hybrid_match

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/match", methods=["POST"])
def match():
    resume = request.form.get("resume", "")
    job_desc = request.form.get("job", "")


    result = hybrid_match(resume, job_desc)

    return render_template("results_nlp.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
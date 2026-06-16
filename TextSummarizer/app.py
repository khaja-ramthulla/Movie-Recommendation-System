
from flask import Flask, render_template, request, jsonify

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

import nltk

# Download required resources once
nltk.download("punkt")
nltk.download("punkt_tab")

app = Flask(__name__)


def get_article_size(word_count):

    if word_count < 300:
        return "Small"

    elif word_count < 1000:
        return "Medium"

    else:
        return "Large"


def generate_summary(text):

    parser = PlaintextParser.from_string(
        text,
        Tokenizer("english")
    )

    summarizer = LsaSummarizer()

    word_count = len(
        text.split()
    )

    # Dynamic number of insights
    sentence_count = min(
        8,
        max(
            3,
            word_count // 100
        )
    )

    summary = summarizer(
        parser.document,
        sentence_count
    )

    result = [
        str(sentence)
        for sentence in summary
    ]

    return result


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/summarize",
    methods=["POST"]
)
def summarize():

    try:

        text = request.form.get(
            "text",
            ""
        ).strip()

        if len(text) < 100:

            return jsonify({
                "error":
                "Please enter at least 100 characters."
            })

        summary = generate_summary(
            text
        )

        original_words = len(
            text.split()
        )

        summary_words = len(
            " ".join(summary).split()
        )

        compression = round(
            (
                1 -
                summary_words /
                original_words
            ) * 100,
            1
        )

        insights_count = len(
            summary
        )

        reading_time_original = round(
            original_words / 200,
            1
        )

        reading_time_summary = round(
            summary_words / 200,
            1
        )

        time_saved = round(
            reading_time_original -
            reading_time_summary,
            1
        )

        quality_score = round(
            min(
                100,
                compression * 1.05
            ),
            1
        )

        article_size = get_article_size(
            original_words
        )

        return jsonify({

            "summary":
            summary,

            "original_words":
            original_words,

            "summary_words":
            summary_words,

            "compression":
            compression,

            "insights_count":
            insights_count,

            "reading_time_original":
            reading_time_original,

            "reading_time_summary":
            reading_time_summary,

            "time_saved":
            time_saved,

            "quality_score":
            quality_score,

            "article_size":
            article_size

        })

    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return jsonify({
            "error":
            str(e)
        })


if __name__ == "__main__":

    app.run(
        debug=True
    )

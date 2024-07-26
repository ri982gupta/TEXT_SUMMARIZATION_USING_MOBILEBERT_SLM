from flask import Flask, render_template, request
import tensorflow as tf
from transformers import T5Tokenizer, TFAutoModelForSeq2SeqLM
from transformers import MobileBertTokenizer
from textblob import TextBlob
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load T5 tokenizer and text generation model for summarization
summarization_tokenizer = T5Tokenizer.from_pretrained('t5-small')
summarization_model = TFAutoModelForSeq2SeqLM.from_pretrained('t5-small')

# Define sentiment text labels
def get_sentiment_text(value):
    if value > 0.75:
        return "Very Positive"
    elif 0.75 >= value > 0.25:
        return "Positive"
    elif 0.25 >= value > -0.25:
        return "Neutral"
    elif -0.25 >= value > -0.75:
        return "Negative"
    else:
        return "Very Negative"

def get_sentiment_polarity(value):
    if value > 0.5:
        return "Positive"
    elif value < -0.5:
        return "Negative"
    else:
        return "Neutral"

def get_sentiment_subjectivity(sentiment_score):
    if sentiment_score > 0.8:
        return "Highly Subjective"
    elif sentiment_score > 0.5:
        return "Moderately Subjective"
    else:
        return "Objective"

# Define sentiment images
def get_sentiment_image(value):
    images = {
        "Very Positive": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAkFBMVE",
        "Positive": "data:image/png;base64BQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQWFvxb/AgZR3MKJMITBAAAAAElFTkSuQmCC",
        "Neutral": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkWLpwprkIs4c0suHpc-d7jW63GQ89OCwVUWf4Fiic9zRrQSGHLmh835rf4Ya0k1LiWss&usqp=CAU",
        "Negative": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxASEhUQEBAWERAXEBgCIiB//Z",
        "Very Negative": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADgCAMAAADCMfHtAAAAh1BMVHi3T5/duNbu7cQXu3iaH9H/+yhBFYgtVZAAAAAElFTkSuQmCC"
    }
    return images.get(get_sentiment_text(value), "url_to_default_image")


# Text summarization function using T5
def summarize_text(text):
    # Tokenize and encode the input text
    inputs = summarization_tokenizer("summarize: " + text, return_tensors="tf", max_length=512, truncation=True)

    # Perform text summarization using T5
    outputs = summarization_model.generate(input_ids=inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the output tokens back to text
    summarized_text = summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summarized_text, text, len(text), len(summarized_text)  # Return summarized text, original text, length of original text, and length of summarized text

def analyze_sentiment(text):
    # Load MobileBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('google/mobilebert-uncased')

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Perform sentiment analysis using MobileBERT
    outputs = model(**inputs)

    # Extract the sentiment score from the model's output
    sentiment_score = outputs.logits[0, 1].item()  # Assuming the output is [batch_size, num_labels]

    return sentiment_score


# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        analysis_type = request.form['analysis_type']
        
        if analysis_type == 'summarization':
            summary, original_txt, len_orig_txt, len_summary = summarize_text(rawtext)
            return render_template('summary.html', summary=summary, original_txt=original_txt, len_orig_txt=len_orig_txt, len_summary=len_summary)

        elif analysis_type == 'sentiment':
            sentiment_score = analyze_sentiment(rawtext)
            sentiment_label = get_sentiment_text(sentiment_score)
            sentiment_polarity = get_sentiment_polarity(sentiment_score)
            sentiment_subjectivity = get_sentiment_subjectivity(sentiment_score)
            return render_template('sentiment.html', original_text=rawtext, sentiment_score=sentiment_score, sentiment_label=sentiment_label, get_sentiment_text=get_sentiment_text, sentiment_polarity=sentiment_polarity,
                           sentiment_subjectivity=sentiment_subjectivity)
        

if __name__ == "__main__":
    app.run(debug=True)

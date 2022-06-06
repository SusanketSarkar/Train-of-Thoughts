from transformers import pipeline
from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer
from flask import Flask,request, url_for, redirect, render_template, jsonify
import numpy as np

app = Flask(__name__,template_folder='templates')
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    genre_provided=request.form.get("genre")
    length_provided=int(request.form.get("len"))
    checkpoint = "rathi/storyGenerator"
    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    input_prompt = "<BOS> <"+genre_provided+">"
    story = story_generator(input_prompt, max_length=length_provided, do_sample=True,
               repetition_penalty=1.1, temperature=1.2, 
               top_p=0.95, top_k=50)
    
    return render_template('home.html',pred='{}'.format(story[0]['generated_text']+'...'))


'''
def predict():
    story_gen = pipeline("text-generation", "pranavpsv/gpt2-genre-story-generator")
    story=story_gen("<BOS> <drama>")
    return render_template('home.html',pred='{}'.format(story))

'''

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['text']
    inputs = tokenizer.encode(user_input, return_tensors='pt')
    outputs = model.generate(inputs, max_length=1000, do_sample=True, top_k=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

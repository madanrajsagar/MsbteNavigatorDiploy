from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from datetime import timedelta
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import mysql.connector as mycon
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_default_secret_key")
app.permanent_session_lifetime = timedelta(days=7)

# Database Connection
mydb = mycon.connect(
    host=os.getenv("DB_HOST", "localhost"), 
    user=os.getenv("DB_USER", "root"), 
    password=os.getenv("DB_PASSWORD", "password"), 
    database=os.getenv("DB_NAME", "signup")
)
db_cur = mydb.cursor()

# Ollama API URL
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Load PDFs and setup QA system
try:
    pdf_directory = "data/"
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    # Summarization
    summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer_pipe = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)

    summarized_texts = [
        summarizer_pipe(text.page_content, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        if len(text.page_content) > 100 else text.page_content for text in texts
    ]

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_texts(summarized_texts, embeddings)

    # Load FLAN-T5 model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device=0 if device == "cuda" else -1
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    db_cur.execute("SELECT * FROM usersdata WHERE username=%s AND password=%s", (username, password))
    result = db_cur.fetchone()
    
    if result:
        session['username'] = username
        flash("Login Successful!", "success")
        return redirect(url_for('index'))
    else:
        flash("Invalid Username or Password!", "danger")
        return redirect(url_for('home'))

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']
    
    if password != confirm_password:
        flash("Passwords do not match!", "danger")
        return redirect(url_for('home'))

    db_cur.execute("SELECT * FROM usersdata WHERE username=%s", (username,))
    if db_cur.fetchone():
        flash("Username already exists!", "danger")
        return redirect(url_for('home'))
    
    db_cur.execute("INSERT INTO usersdata (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
    mydb.commit()
    flash("Signup successful! You can now log in.", "success")
    return redirect(url_for('home'))

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('index.html', username=session['username'])

@app.route('/ask', methods=['POST'])
def ask():
    if 'username' not in session:
        return jsonify({"answer": "Please log in.", "status": "error"})
    
    data = request.get_json()
    question = data.get("question", "").strip()
    pdf_answer = qa.run(question)

    if len(pdf_answer.split()) < 15 or "I don't know" in pdf_answer.lower():
        response = requests.post(OLLAMA_URL, json={"model": "mistral", "prompt": question, "stream": False})
        return jsonify({"answer": response.json().get("response", ""), "status": "success"})
    
    return jsonify({"answer": pdf_answer, "status": "success"})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

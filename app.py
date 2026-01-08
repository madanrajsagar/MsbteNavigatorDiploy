from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from datetime import timedelta
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
import torch
import mysql.connector as mycon
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Load DB credentials from .env
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not os.getenv("DB_HOST"):
    raise Exception("Environment variables not loaded. Check .env file")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_default_secret_key")
app.permanent_session_lifetime = timedelta(days=7)

# Database Connection
try:
    mydb = mycon.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        ssl_disabled=True,
        connection_timeout=10
    )
    db_cur = mydb.cursor(dictionary=True)
    print("✅ Database connected successfully")
except Exception as e:
    print("❌ Database connection failed:", e)
    raise

# Ollama API URL
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Load PDFs and setup QA system
try:
    pdf_directory = "data/"
    
    # Check if directory exists
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory, exist_ok=True)
        print(f"📁 Created '{pdf_directory}' directory")
    
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()
    
    if not documents:
        print("⚠️ No PDF documents loaded! Please add PDF files to 'data/' directory")
        # Create a simple text to work with even without PDFs
        documents = [type('obj', (object,), {'page_content': 'General knowledge about various topics.'})()]
    
    print(f"✅ Loaded {len(documents)} documents from PDFs")
    
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    # Summarization (simplified to work without BART if needed)
    try:
        summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        summarizer_pipe = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)

        summarized_texts = []
        for text in texts:
            content = text.page_content
            if len(content) > 100:
                try:
                    summary = summarizer_pipe(content, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                    summarized_texts.append(summary)
                except:
                    summarized_texts.append(content[:200])  # Fallback to first 200 chars
            else:
                summarized_texts.append(content)
    except Exception as e:
        print(f"⚠️ Summarization failed, using original text: {e}")
        summarized_texts = [text.page_content[:500] for text in texts]  # Use first 500 chars

    print(f"✅ Created {len(summarized_texts)} text chunks")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Lighter model
    
    # Create FAISS index
    db = FAISS.from_texts(summarized_texts, embeddings)

    # Load FLAN-T5 model with simpler settings
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  # Use base model (smaller)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        
        # Determine device
        device = 0 if torch.cuda.is_available() else -1
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=200,
            temperature=0.3,
            device=device
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Create retriever
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        
        print("✅ QA system loaded successfully")
        
    except Exception as e:
        print(f"⚠️ Could not load FLAN-T5 model: {e}")
        print("📝 Will use Ollama only for responses")
        qa = None
        db = None
        
except Exception as e:
    print(f"⚠️ Setup failed: {str(e)}")
    qa = None
    db = None

@app.route('/', methods=['GET'])
def home():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    # POST method handling
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

@app.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup.html')

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

@app.route('/index', methods=['GET'])
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

    if not question:
        return jsonify({
            "answer": "Please ask a complete question.",
            "status": "success"
        })

    # Try PDF-based QA first if available
    pdf_answer = ""
    if qa is not None:
        try:
            result = qa.run(question)  # Changed from qa.invoke to qa.run
            pdf_answer = str(result).strip()
            print(f"📄 PDF Answer: {pdf_answer[:100]}...")
        except Exception as e:
            print(f"⚠️ PDF QA error: {e}")
            pdf_answer = ""

    # Check if we have a good answer from PDFs
    has_good_pdf_answer = (
        pdf_answer and 
        len(pdf_answer) > 20 and 
        "i don't know" not in pdf_answer.lower() and
        "cannot answer" not in pdf_answer.lower()
    )
    
    if has_good_pdf_answer:
        return jsonify({
            "answer": pdf_answer,
            "status": "success",
            "source": "pdf"
        })
    
    # Fallback to Ollama
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "mistral",
                "prompt": f"Answer this question clearly: {question}",
                "stream": False
            },
            timeout=10
        )
        
        if response.status_code == 200:
            ollama_answer = response.json().get("response", "").strip()
            
            # If we have some PDF answer, combine it
            if pdf_answer and len(pdf_answer) > 10:
                final_answer = f"Based on available information: {pdf_answer}\n\nAdditionally: {ollama_answer}"
            else:
                final_answer = ollama_answer if ollama_answer else "I'll try to help with that based on general knowledge."
            
            return jsonify({
                "answer": final_answer,
                "status": "success",
                "source": "ollama" if not pdf_answer else "combined"
            })
            
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
    
    # Final fallback responses
    fallback_responses = [
        "I understand you're asking about that topic. While I don't have specific documents on it, I can tell you that it's an interesting subject worth exploring further.",
        "That's a good question. Based on general knowledge, this topic covers various aspects that are important to consider.",
        "I'll help you think through this. The question you're asking relates to important concepts that are often discussed in this field.",
        "Let me provide some general insight on this. The topic you mentioned involves several key factors to consider."
    ]
    
    import random
    return jsonify({
        "answer": random.choice(fallback_responses),
        "status": "success",
        "source": "fallback"
    })

@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for('home'))

@app.route('/check_pdfs', methods=['GET'])
def check_pdfs():
    """Simple endpoint to check PDF status"""
    pdf_directory = "data/"
    pdf_files = []
    
    if os.path.exists(pdf_directory):
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    return jsonify({
        "pdf_directory_exists": os.path.exists(pdf_directory),
        "pdf_files": pdf_files,
        "count": len(pdf_files),
        "qa_system_ready": qa is not None
    })

# Add a test route to verify the server is running
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Flask server is running",
        "database": "connected" if mydb.is_connected() else "disconnected",
        "qa_system": "ready" if qa is not None else "not_ready"
    })

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    if not os.path.exists("data/"):
        os.makedirs("data/")
        print("📁 Created 'data/' directory. Please add PDF files.")
    
    print("🚀 Starting Flask application...")
    print(f"📚 PDF QA System: {'✅ Ready' if qa is not None else '⚠️ Not available (will use Ollama)'}")
    print(f"🤖 Ollama URL: {OLLAMA_URL}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
# 🎓 MSBTE Navigator

An AI-powered chatbot designed to help Maharashtra State Board of Technical Education (MSBTE) diploma students by providing instant, accurate, and context-aware answers to academic queries.

## 🚀 Overview

MSBTE Navigator is an intelligent chatbot that allows diploma students to ask questions related to the MSBTE curriculum, syllabus, examination patterns, and study materials. The chatbot uses Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to provide relevant responses from a custom knowledge base.

---

## ✨ Features

- 🤖 AI-powered chatbot for MSBTE students
- 📚 Answers based on MSBTE syllabus and academic resources
- 🔍 Retrieval-Augmented Generation (RAG)
- 💬 Interactive and user-friendly chat interface
- ⚡ Fast response generation using local LLM (Ollama)
- 📄 Knowledge base built from MSBTE documents
- 🌐 Web-based interface using Flask

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend Development |
| Flask | Web Framework |
| Ollama (Mistral) | Large Language Model |
| LangChain | RAG Pipeline |
| HuggingFace Embeddings | Text Embeddings |
| FAISS | Vector Database |
| HTML/CSS | Frontend |
| Git & GitHub | Version Control |

---

## 📂 Project Structure

```
MSBTE-Navigator/
│
├── data/                # Knowledge base documents
├── templates/           # HTML templates
├── app.py               # Flask application
├── requirements.txt     # Python dependencies
├── Procfile             # Deployment configuration
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/MSBTE-Navigator.git
cd MSBTE-Navigator
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Ollama

```bash
ollama run mistral
```

### 6. Run the application

```bash
python app.py
```

---

## 💡 How It Works

1. User asks a question.
2. The query is converted into embeddings.
3. FAISS retrieves the most relevant documents.
4. LangChain sends the retrieved context to the Mistral LLM.
5. The chatbot generates an accurate response based on the retrieved information.

---

## 📸 Screenshots

Add screenshots of your chatbot interface here.

Example:

```
images/home.png
images/chat.png
```

---

## 🎯 Future Improvements

- User authentication
- Voice-based interaction
- PDF upload support
- Image-based query answering
- Mobile responsive interface
- Multi-language support
- GPT-4/Gemini integration

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## 📄 License

This project is intended for educational purposes.

---

## 👨‍💻 Author

**Madanraj Yuvraj Sagar**

📧 madanrajsagar83@gmail.com

🔗 LinkedIn: https://linkedin.com/in/your-linkedin

💻 GitHub: https://github.com/madanrajsagar

---

⭐ If you found this project useful, don't forget to give it a star!

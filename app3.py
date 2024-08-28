from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

app = Flask(__name__)

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = ""

# Load PDF and process text
def process_pdf(pdf_path):
    pdfreader = PdfReader(pdf_path)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    text = text_splitter.split_text(raw_text)
    
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(text, embeddings)
    
    return document_search

# Initialize the LangChain pipeline
chain = load_qa_chain(OpenAI(), chain_type='stuff')

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        query = request.form['query']
        pdf_path = "sample_report-pages.pdf"  # You can also allow users to upload PDFs
        document_search = process_pdf(pdf_path)
        docs = document_search.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
    return render_template('index3.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True)

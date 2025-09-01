import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

class DreamKnowledgeBase:
    def __init__(self, pdf_directory: str, vector_directory: str):
        self.pdf_directory = pdf_directory
        self.vector_directory = vector_directory
        self.vectorstore = None
        self.embeddings = OpenAIEmbeddings()
        
    async def initialize(self):
        """Load existing vectors or create new ones."""
        if os.path.exists(os.path.join(self.vector_directory, "index.faiss")):
            print("Loading existing knowledge base...")
            self.vectorstore = FAISS.load_local(
                self.vector_directory, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("Building knowledge base from files...")
            await self.build_knowledge_base()
    
    async def build_knowledge_base(self):
        """Process PDFs and text files to create FAISS index."""
        if not os.path.exists(self.pdf_directory):
            print(f"Knowledge directory {self.pdf_directory} not found")
            return
            
        documents = []
        
        # Load PDFs
        pdf_loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        
        # Load text files
        txt_loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf8'}
        )
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        
        if not documents:
            print("No PDFs or text files found in knowledge base directory")
            return
            
        print(f"Found {len(pdf_docs)} PDFs and {len(txt_docs)} text files")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create FAISS index
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Save to disk
        os.makedirs(self.vector_directory, exist_ok=True)
        self.vectorstore.save_local(self.vector_directory)
        print(f"Knowledge base created with {len(chunks)} chunks from {len(documents)} files")
    
    async def search_relevant_knowledge(self, query: str, k: int = 3) -> List[Document]:
        """Search for relevant passages."""
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)
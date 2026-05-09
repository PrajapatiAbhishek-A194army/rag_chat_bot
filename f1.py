import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi


# Load Environment Variables
load_dotenv()


class RAGSystem:

    def __init__(self, data_path=None, index_path=None):

        # Base Directory
        self.base_dir = Path(__file__).parent.parent

        # Paths
        self.data_path = data_path or (
            self.base_dir /
            "data" /
            "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
        )

        self.index_path = index_path or (
            self.base_dir / 'model'/"faiss_index"
        )

        # Components
        self.embedding_model = None
        self.vector_db = None
        self.bm25 = None
        self.corpus = None
        self.reranker = None
        self.llm = None
        self.df = None

        # Initialize System
        self.initialize()

    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------

    def initialize(self):

        print("\nInitializing RAG System...\n")

        # ---------------------------------------------------
        # 1. LOAD DATA
        # ---------------------------------------------------

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        self.df = pd.read_csv(self.data_path)

        # Clean Data
        self.df = (
            self.df[['intent', 'instruction', 'response']]
            .dropna()
            .drop_duplicates()
        )

        self.df = self.df[
            self.df['response'].str.len() > 20
        ]

        print(f"Dataset Loaded: {len(self.df)} rows")

        # ---------------------------------------------------
        # 2. CREATE DOCUMENTS
        # ---------------------------------------------------

        documents = []

        for row in self.df.itertuples():

            text = f"""
                Intent: {row.intent}

                Question:
                {row.instruction}

                Answer:
                {row.response}
                """

            documents.append(text)

        # ---------------------------------------------------
        # 3. TEXT CHUNKING
        # ---------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )

        docs = splitter.create_documents(documents)

        print(f"Chunks Created: {len(docs)}")

        # ---------------------------------------------------
        # 4. EMBEDDING MODEL
        # ---------------------------------------------------

        print("\nLoading Embedding Model...\n")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ---------------------------------------------------
        # 5. FAISS VECTOR DATABASE
        # ---------------------------------------------------

        faiss_file = self.index_path / "index.faiss"

        if faiss_file.exists():

            print("Loading Existing FAISS Index...\n")

            self.vector_db = FAISS.load_local(
                str(self.index_path),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )

        else:

            print("Creating New FAISS Index...\n")

            self.vector_db = FAISS.from_documents(
                docs,
                self.embedding_model
            )

            self.vector_db.save_local(
                str(self.index_path)
            )

            print("FAISS Index Created Successfully\n")

        # ---------------------------------------------------
        # 6. BM25 KEYWORD SEARCH
        # ---------------------------------------------------

        print("Initializing BM25...\n")

        self.corpus = [
            doc.page_content
            for doc in docs
        ]

        tokenized_corpus = [
            doc.split()
            for doc in self.corpus
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        # ---------------------------------------------------
        # 7. RERANKER MODEL
        # ---------------------------------------------------

        print("Loading Reranker Model...\n")

        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )

        # ---------------------------------------------------
        # 8. GROQ LLM
        # ---------------------------------------------------

        print("Connecting to Groq LLM...\n")

        try:

            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0
            )

            print("Groq LLM Connected Successfully\n")

        except Exception as e:

            self.llm = None

            print(f"Groq Connection Failed: {e}")

    # ---------------------------------------------------
    # MULTI QUERY GENERATION
    # ---------------------------------------------------

    def generate_queries(self, query):

        return [
            query,
            f"{query} refund issue",
            f"{query} payment issue",
            f"{query} explanation",
        ]

    # ---------------------------------------------------
    # HYBRID SEARCH
    # ---------------------------------------------------

    def hybrid_search(self, query, k=5):

        # Vector Search
        vector_results = self.vector_db.similarity_search(
            query,
            k=k
        )

        # BM25 Search
        try:

            keyword_results = self.bm25.get_top_n(
                query.split(),
                self.corpus,
                n=k
            )

        except Exception:

            keyword_results = []

        # Combine Results
        combined = [
            doc.page_content
            for doc in vector_results
        ]

        combined.extend(keyword_results)

        return list(set(combined))

    # ---------------------------------------------------
    # RERANK RESULTS
    # ---------------------------------------------------

    def rerank(self, query, docs, top_k=3):

        pairs = [
            (query, doc)
            for doc in docs
        ]

        scores = self.reranker.predict(pairs)

        ranked_docs = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, score in ranked_docs[:top_k]
        ]

    # ---------------------------------------------------
    # MAIN QUERY FUNCTION
    # ---------------------------------------------------

    def query(self, user_query):

        # Generate Multi Queries
        queries = self.generate_queries(user_query)

        all_docs = []

        # Retrieve Docs
        for q in queries:

            docs = self.hybrid_search(q, k=5)

            all_docs.extend(docs)

        # Remove Duplicates
        all_docs = list(set(all_docs))

        # Rerank Docs
        top_docs = self.rerank(
            user_query,
            all_docs,
            top_k=3
        )

        # Create Context
        context = "\n\n".join(top_docs)

        # Prompt
      
        prompt = f"""
You are a helpful and professional E-Commerce Customer Support Assistant.

Your task is to answer the user's question ONLY using the provided context.

Guidelines:
- Provide clear, concise, and accurate answers.
- Be polite and customer-friendly.
- Do not make up information.
- If the answer is not present in the context, respond with:
  "I don't know based on the provided information."

Retrieved Context:
{context}

User Question:
{user_query}

Helpful Answer:
"""
        # Generate Response
        if self.llm:

            response = self.llm.invoke(prompt)

            return response.content, top_docs

        else:

            return "Groq API Connection Failed", top_docs


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == "__main__":

    rag = RAGSystem()

    print("\nChatbot Ready")
    print("Type 'exit' to quit\n")
    # conversation=''
    while True:

        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        answer, sources = rag.query(user_input)
        # conversation+='\n bot:'+user_input+'\n--------------------------------------------------------------\n'
        print("\nBot:")
        print(answer)

        # print("\nRetrieved Sources:")
        # for i, source in enumerate(sources, start=1):
        #     print(f"\nSource {i}:")
        #     print(source[:300])
        #     print("-" * 50)


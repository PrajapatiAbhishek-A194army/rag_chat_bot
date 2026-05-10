# AI-Powered RAG Customer Support Chatbot

## Overview

This project is an AI-powered Customer Support Chatbot built using Retrieval-Augmented Generation (RAG). The chatbot can answer customer queries related to refunds, payments, cancellations, and order tracking using semantic search and Large Language Models (LLMs).

## Features

* Hybrid Search using FAISS and BM25
* Semantic Retrieval with Hugging Face Embeddings
* CrossEncoder Reranking for better response accuracy
* Groq LLM integration for fast AI-generated responses
* Context-aware and accurate customer support answers

## Tech Stack

Python, LangChain, FAISS, Hugging Face, Sentence Transformers, BM25, Groq API, Pandas

## Project Workflow

1. Load and preprocess customer support dataset
2. Create embeddings and store them in FAISS
3. Retrieve relevant documents using hybrid search
4. Rerank retrieved results using CrossEncoder
5. Generate final response using Groq LLM

## Use Cases

* E-Commerce Customer Support
* AI Chatbots
* FAQ Assistance
* RAG-based NLP Applications

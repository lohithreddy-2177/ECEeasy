🎓 ECEEasy - ECE Course Assistant

A sophisticated Retrieval-Augmented Generation (RAG) system designed to help Electrical and Computer Engineering (ECE) students explore course information, reviews, and insights using natural language queries

✨ Features

Natural Language Queries: Ask questions about courses in plain English
Course-Specific Filtering: Automatically detects course names from queries for precise retrieval
Comprehensive Course Information: Access details on instructors, difficulty, usefulness, challenges, reviews, and more
Smart Context Management: Advanced chunking strategies with optimal overlap for better context retention
Local LLM Integration: Runs entirely offline using Ollama with Llama3 model
Vector Search: Efficient semantic search using ChromaDB and embeddings
Student Review Analysis: Aggregates and synthesizes multiple student perspectives

📋 Supported Queries.
Ask about any aspect of ECE courses:
"Who teaches Digital Signal Processing?"
"What are the challenges in RF Simulation Techniques?"
"How useful is the VLSI Design course?"
"Summarize the Design of Passive Microwave Components course"
"What are the prerequisites for Fundamentals of Communication System?"
"Tell me about student reviews for Embedded Systems Design"

🛠️ Technical Architecture
~~~
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ Course Matcher  │───▶│  Vector Store   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                       │
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAG Chain     │◀───│ Context Filter  │◀───│ Similarity      │
│                 │    │                 │    │ Search          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼        
┌─────────────────┐
│  LLM (Llama3)   │
│  + Synthesis    │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Comprehensive  |
│     Answer      │
└─────────────────┘
~~~

📁 Project Structure
eceeasy_advanced/
├── eceeasy_advanced.py     # Main application
├── courses_data/           # Course JSON files
│   └── ece_courses.json   # Sample course data
├── chroma_db/             # Vector database (auto-generated)
├── requirements.txt       # Python dependencies
└── README.md             # This file

Prerequisites
Python 3.8+
Ollama installed locally
Llama3 model downloaded in Ollama

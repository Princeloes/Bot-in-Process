# AI Ticket Support System - Internship Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Princeloes/Bot-in-Process/blob/main/ticket_grouping.ipynb)

## Goal
Design an AI system to automatically group customer support tickets and generate appropriate responses using NLP and RAG.

## Features
1. **Ticket Grouping** (Classification/Clustering) using Embeddings.
2. **Automated Responses** using Retrieval-Augmented Generation (RAG).

## Architecture
- **Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Classification Strategy**: Zero-Shot or Clustering (K-Means)
- **Response Generation**: LLM + Knowledge Base

## Roadmap
- [ ] Phase 1: System Design & Grouping Strategy
- [ ] Phase 2: Technical Implementation (Embeddings & Classifier)
- [ ] Phase 3: Prototype & Testing

## How to Run (Planned)
1. Install dependencies: `pip install -r requirements.txt`
2. Run grouping script: `python group_tickets.py`

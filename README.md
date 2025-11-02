# TattvaShanti Chatbot (AI-Powered Conversational Assistant)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-Framework-orange)
![LangChain](https://img.shields.io/badge/LangChain-LLM%20Pipeline-purple)
![OpenAI](https://img.shields.io/badge/OpenAI-API-blue)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-teal)
![n8n](https://img.shields.io/badge/n8n-Automation-red)
![Render](https://img.shields.io/badge/Render-Deployment-blue)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-driven **Conversational RAG Chatbot** designed and deployed for [Tattva Shanti](https://tattvashanti.com/). an organisation offering a range of wellness and business support services, while this chatbot is trained specifically on their **Life Coaching**, **Professional/Startup Coaching** and **EIR (Entrepreneur-in-Residence)** tracks.

This project combines **FastAPI**, **LangChain**, **OpenAI**, **Pinecone**, and **n8n automation** to deliver an intelligent, context-aware, and production-ready chatbot embedded directly into the WordPress website.

---

## Overview

The **Tattva Shanti Chatbot** acts as a virtual assistant that:
- Understands user intent and provides conversational, human-like responses.
- Restricts its domain knowledge to three key topics:
  1. Life Coaching  
  2. Professional/Startup Coaching  
  3. Entrepreneur-in-Residence (EIR) Program  
- Integrates AI retrieval from curated and embedded content.
- Automates data logging to Google Sheets via n8n workflows.

From data to deployment: this system uses RAG techniques, maintains user session context, and integrates directly into the WordPress environment.

---

## Core AI Architecture

### 1. **Retrieval-Augmented Generation (RAG)**
- Cleaned and structured domain-specific data (scraped & manually enriched with metadata).
- Embedded text using **OpenAI’s `text-embedding-3-small`** model.
- Stored vector representations in **Pinecone Vector Database**.
- Real-time retrieval of top-k contextually relevant documents per query.

### 2. **Generative Layer**
- Used **OpenAI’s `gpt-3.5-turbo`** for controlled, domain-constrained response generation.
- A **custom system prompt** ensures:
  - Warm, professional tone.
  - Context-limited knowledge.
  - Boundary management for irrelevant or restricted topics (e.g., medical, contact info).

### 3. **Automated Logging & Analytics**
- Every query-response pair is **sent via webhook to n8n**, which:
  - Logs chat data into **Google Sheets** for analytics.
  - Enables further automation, such as feedback tracking and engagement metrics.

---

## AI Features

| Feature | Description |
|----------|-------------|
| **RAG Pipeline** | Real-time retrieval + generation using OpenAI and Pinecone. |
| **Prompt Engineering** | Controlled response rules for professional tone, topic restriction, and ethical handling. |
| **Session Memory** | Maintains user chat context with short-term memory (last 20 turns). |
| **Data Validation** | NLP-based question checks (length, relevance, alpha ratio) to prevent prompt injection. |
| **Logging Automation** | Auto-logs all interactions into Google Sheets via n8n webhook. |
| **Error Recovery** | Timeout handling, fallback responses, and user-friendly error prompts. |
| **CORS-secured Backend** | Only accessible from `https://tattvashanti.com` for secure frontend communication. |
| **Scalable Deployment** | Hosted on **Render**, ensuring uptime and automatic session cleanup. |

---

## Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | HTML5, CSS3 (custom chat widget for WordPress embedding) |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) |
| **LLM Integration** | [LangChain](https://www.langchain.com/) + [OpenAI GPT-3.5 Turbo](https://platform.openai.com/docs/models/gpt-3-5) |
| **Vector DB** | [Pinecone](https://www.pinecone.io/) |
| **Automation** | [n8n](https://n8n.io/?ps_partner_key=YzQ1MWQxYjZjNDgx&ps_xid=3SG8GJIcNIvm5S&gsxid=3SG8GJIcNIvm5S&gspk=YzQ1MWQxYjZjNDgx&gad_campaignid=23194935516) (for webhook-based Google Sheets logging) |
| **Deployment** | [Render Cloud Platform](https://render.com/) |
| **Environment Management** | Python-dotenv, UUID, asyncio |

---

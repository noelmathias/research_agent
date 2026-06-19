# Autonomous Multi-Agent Research Assistant

An end-to-end Agentic AI platform that automates the research workflow using specialized AI agents, Retrieval-Augmented Generation (RAG), semantic search, and evaluator-guided reflection loops. 

The system ingests documents, performs semantic retrieval, synthesizes research findings, evaluates response quality, and generates citation-backed reports through a multi-agent orchestration framework powered by **LangGraph**.

---
## 🔍 Overview

This project implements an autonomous research pipeline consisting of specialized AI agents that collaborate to produce high-quality, comprehensive research reports. By shifting from standard single-prompt RAG to a **graph-based multi-agent workflow**, the system breaks down complex queries, cross-references sources, self-evaluates for hallucinations, and refines its output autonomously before presenting it to the user.

## 📑 Table of Contents
- [Overview](#-overview)
- [🏗️ System Architecture](#️-system-architecture)
- [🤖 Agent Workflow & Core Pipeline](#-agent-workflow--core-pipeline)
- [✨ Key Features](#-key-features)
- [💻 Tech Stack](#-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🛠️ Step-by-Step Implementation Guide](#️-step-by-step-implementation-guide)
- [🚀 Local Setup & Execution](#-local-setup--execution)
- [🐳 Docker Deployment](#-docker-deployment)
- [🎯 Example Use Cases](#-example-use-cases)
- [🔮 Future Enhancements](#-future-enhancements)
- [💼 Resume Highlights](#-resume-highlights)
- [📝 License](#-license)
- [👤 Author](#-author)

---
## 🏗️ System Architecture

![Autonomous Multi-Agent Research Assistant Refined System Architecture](assets/architecture.png)

---

## 🤖 Agent Workflow & Core Pipeline

The platform uses a stateful, cyclic graph managed by **LangGraph** to pass context seamlessly between specialized agents. Rather than running a single linear prompt, the system routes state transitions through a structured multi-agent loop:

```text
Planner ──> Retriever ──> Summarizer ──> Evaluator ──> Reflection ──> Report Generator
```
---

## 📋 Agent Responsibilities (The 6-Stage Deep Dive)

 **1. Planner Agent**
   * Decomposes user queries into structured research objectives.
   * Generates a research plan and retrieval strategy.

 **2. Retriever Agent**
   * Performs semantic search over ChromaDB.
   * Retrieves the most relevant context and source metadata.

 **3. Summarizer Agent**
   * Synthesizes retrieved information into a coherent draft.
   * Preserves source grounding and citations.

 **4. Evaluator Agent**
   * Assesses factual consistency and response quality.
   * Produces confidence scores and validation feedback.

 **5. Reflection Agent**
   * Uses evaluator feedback to identify missing context.
   * Triggers retrieval retries and plan refinement when needed.

 **6. Report Agent**
   * Generates the final citation-backed research report.
   * Formats outputs into publication-ready Markdown.

---

## ✨ Key Features

### 🤝 Multi-Agent Orchestration
* LangGraph-based stateful execution
* Conditional routing and reflection loops
* Structured agent-to-agent state transitions

### 📚 Retrieval-Augmented Generation (RAG)
* ChromaDB vector storage
* Semantic similarity search
* Citation-aware response generation

### 📄 Document Processing
* PDF ingestion with PyMuPDF
* Metadata-aware chunking and indexing
* Persistent vector storage

### 🛠️ Reliability & Monitoring
* Confidence-based evaluation
* Reflection-driven self-correction
* Structured logging and execution tracking

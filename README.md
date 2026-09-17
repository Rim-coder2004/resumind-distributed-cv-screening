# Resumind — Distributed CV Screening Platform

Resumind is a distributed CV screening platform that combines asynchronous processing, machine learning and semantic embeddings to help recruiters evaluate candidates against specific job descriptions.

The project was developed as a Big Data / Data Science project and brings together a web application, Apache Kafka, Apache Spark MLlib, sentence embeddings, Mistral AI and PostgreSQL.

## 🎯 Project Overview

The platform is designed around a complete recruitment workflow:

1. A recruiter creates a job posting with a title, description and required skills.
2. A candidate submits a CV in PDF format.
3. The CV is processed asynchronously through Apache Kafka.
4. Mistral AI performs OCR when required and extracts structured candidate information.
5. Apache Spark MLlib provides a CV-based classification score using a Random Forest model.
6. `all-MiniLM-L6-v2` embeddings compare the candidate profile with the target job description using cosine similarity.
7. The two signals are combined into a final score and mapped to one of three decisions:
   - `shortlist`
   - `review`
   - `reject`
8. Mistral AI generates tailored interview questions for the candidate and role.
9. Candidate information, scores, decision and interview questions are stored in PostgreSQL.

## 🏗️ Architecture

```text
Recruiter / Web Application
          │
          ▼
     Node.js / Express
          │
          ▼
   Apache Kafka
   ┌───────────────┐
   │ cv-submissions│
   └───────┬───────┘
           │
           ▼
    Python Consumer
           │
     ┌─────┴──────────────┐
     │                    │
     ▼                    ▼
Mistral AI            Spark MLlib
OCR + extraction      Random Forest
     │                    │
     └─────────┬──────────┘
               ▼
        MiniLM Embeddings
        JD / CV similarity
               │
               ▼
        Final Decision
   shortlist / review / reject
               │
               ▼
          PostgreSQL
               │
               ▼
        Recruiter Dashboard
```

## 🧠 AI & Machine Learning

### Random Forest — Spark MLlib

A Random Forest classifier is used to produce a CV-based score using features such as:

- Number of skills
- Years of experience
- Education information

### Semantic Matching — MiniLM

The platform also uses `all-MiniLM-L6-v2` from Sentence Transformers.

The model converts the job description and candidate profile into 384-dimensional embeddings. Cosine similarity is then used to measure semantic alignment between the candidate and the target role.

This adds job-specific context that a CV-only classifier cannot provide.

### Mistral AI

Mistral AI is used for:

- OCR of scanned CV documents
- Structured CV information extraction
- Generation of tailored interview questions

## ⚙️ Technology Stack

| Area | Technologies |
|---|---|
| Frontend | React |
| Backend | Node.js, Express |
| Messaging | Apache Kafka, KafkaJS, kafka-python |
| Big Data / ML | Apache Spark, Spark MLlib |
| Semantic AI | Sentence Transformers, MiniLM-L6-v2 |
| Generative AI / OCR | Mistral AI |
| Database | PostgreSQL / Supabase |
| ORM | Prisma |
| Main language | Python, JavaScript |
| Infrastructure | Kafka, PostgreSQL |

## 📂 Repository Structure

```text
resumind-distributed-cv-screening/
│
├── README.md
├── src/
│   └── cv_consumer.py
├── notebooks/
│   └── spark_train_model.ipynb
└── docs/
    ├── report.docx
    └── presentation.pptx
```

> The repository contains the main data-processing and machine-learning materials from the project. The complete web application/backend/frontend source code is not included in the currently prepared project archive.

## 🔄 Processing Pipeline

### 1. CV ingestion

The system receives a candidate CV and publishes the candidate data to the Kafka topic `cv-submissions`.

### 2. CV extraction

The Python consumer receives the message and processes the candidate information. Mistral AI can perform OCR and structured field extraction.

### 3. CV scoring

Spark MLlib loads the trained Random Forest model and produces a CV-based score.

### 4. Job-description matching

The candidate profile and job description are encoded with MiniLM. Cosine similarity produces a semantic job-match score.

### 5. Score blending

The consumer combines the CV score and job-match score using configurable weights.

The implementation exposes environment variables for:

```text
SHORTLIST_THRESHOLD
REJECT_THRESHOLD
CV_SCORE_WEIGHT
```

This makes the decision thresholds configurable without modifying the source code.

### 6. Interview question generation

For candidates classified as `shortlist` or `review`, Mistral AI generates five role-specific interview questions using the job description and candidate profile.

### 7. Persistence

The candidate record is stored in PostgreSQL together with the scoring information, final decision and generated questions.

## 📊 Example Decision Flow

```text
CV
 │
 ├── Candidate features ──► Spark Random Forest ──► CV score
 │
 └── Candidate profile
             │
             ├──────────────► MiniLM
             │                  │
Job Description ────────────────┘
                                ▼
                         JD match score
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
             Score blending            Decision rules
                                             │
                         ┌───────────────────┼──────────────────┐
                         ▼                   ▼                  ▼
                      shortlist            review             reject
```

## 📚 Project Materials

- `src/cv_consumer.py` — Python Kafka consumer and scoring pipeline
- `notebooks/spark_train_model.ipynb` — Spark model training notebook
- `docs/report.docx` — project report
- `docs/presentation.pptx` — project presentation

## 🔐 Security

This public repository should never contain:

- API keys
- Database passwords
- Kafka credentials
- OAuth tokens
- `.env` files
- Private database connection strings
- Production credentials

Use environment variables for secrets and configuration.

Before publishing, verify that no credentials or personal candidate information are present in notebooks, documents, source files or exported datasets.

## 🚀 Future Improvements

Potential extensions described for the project include:

- Adding a real-time analytics dashboard using Kafka
- Adapting decision thresholds based on recruiter feedback
- Supporting simultaneous submission of multiple CVs with comparative ranking
- Containerising the complete platform
- Deploying the complete system publicly

## 🎓 Project Context

Resumind was developed as a practical Big Data / Data Science project to apply distributed messaging, machine learning and semantic text processing to an end-to-end recruitment use case.

## 👤 Author

**Rim Akarche**

GitHub: [Rim-coder2004](https://github.com/Rim-coder2004)

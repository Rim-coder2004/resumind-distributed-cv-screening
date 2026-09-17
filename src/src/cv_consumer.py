import json
import re
import os
import psycopg2
import requests
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def format_questions(questions):
    if not questions or not isinstance(questions, list) or len(questions) == 0:
        return None  # store NULL when no questions
    return '\n\n'.join(f"{i+1}. {q}" for i, q in enumerate(questions))


# ─── CONFIG ────────────────────────────────────────────────
KAFKA_BROKER    = 'localhost:9092'
TOPIC_IN        = 'cv-submissions'
TOPIC_OUT       = 'scored-candidates'
MODEL_PATH      = './cv_spark_model'
LABEL_MAP       = {0.0: "shortlist", 1.0: "review", 2.0: "reject"}
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY', 'YOUR_MISTRAL_KEY_HERE')
DATABASE_URL    = os.environ.get('DATABASE_URL')

# Tunable thresholds (env vars so we can adjust live without restarting code edits)
SHORTLIST_THRESHOLD = float(os.environ.get('SHORTLIST_THRESHOLD', '0.65'))
REJECT_THRESHOLD    = float(os.environ.get('REJECT_THRESHOLD',    '0.40'))
CV_SCORE_WEIGHT     = float(os.environ.get('CV_SCORE_WEIGHT',     '0.30'))
JD_MATCH_WEIGHT     = 1.0 - CV_SCORE_WEIGHT

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

print(f"⚙️  Thresholds → shortlist ≥ {SHORTLIST_THRESHOLD}, reject < {REJECT_THRESHOLD}")
print(f"⚙️  Score blend → {CV_SCORE_WEIGHT:.2f} × cv + {JD_MATCH_WEIGHT:.2f} × jd_match")


# ─── SPARK + MODEL ─────────────────────────────────────────
print("🔥 Starting Spark session...")
spark = SparkSession.builder.appName("CVConsumer").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print(f"📦 Loading Spark model from {MODEL_PATH}...")
model = RandomForestClassificationModel.load(MODEL_PATH)
print("✅ Spark model loaded")


# ─── EMBEDDING MODEL ───────────────────────────────────────
print("🧠 Loading sentence-transformer (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded")


# ─── POSTGRES CONNECTION ──────────────────────────────────
print("🐘 Connecting to Postgres...")
pg_conn = psycopg2.connect(DATABASE_URL)
pg_conn.autocommit = True
print("✅ Postgres connected")


# ─── FEATURE ENGINEERING (Spark RF) ────────────────────────
def featurize(skills, years_str, education):
    skills_count = len([s for s in re.split(r'[,\n]', skills or '') if s.strip()])

    years_match = re.search(r'\d+', str(years_str or '0'))
    years = int(years_match.group()) if years_match else 0
    if years > 100:  # treat as a year e.g. 2019
        years = max(0, 2024 - years)

    has_degree = 1 if (education or '').strip() else 0
    return skills_count, years, has_degree


# ─── JD-AWARE EMBEDDING SIMILARITY ─────────────────────────
def compute_jd_match_score(candidate, jd_text, required_skills=''):
    """
    Embeds the JD text and a candidate profile blob, returns cosine similarity in [0, 1].
    Higher = better semantic match between candidate and JD.
    """
    # Build a candidate profile string that captures their full picture
    candidate_blob = " ".join(filter(None, [
        f"Skills: {candidate.get('skills', '')}",
        f"Experience: {candidate.get('years_of_experience', 0)} years",
        f"Education: {candidate.get('education', '')}"
    ]))

    # Build a JD blob — fold required_skills in if present, since they're high-signal
    jd_blob = jd_text or ''
    if required_skills:
        jd_blob = f"{jd_blob}\n\nKey required skills: {required_skills}"

    if not candidate_blob.strip() or not jd_blob.strip():
        return 0.0

    # Encode both, compute cosine similarity
    embeddings = embed_model.encode([jd_blob, candidate_blob])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Cosine sim is in [-1, 1] but for text from MiniLM it's effectively in [0, 1].
    # Clip to [0, 1] just in case to keep the blend math well-behaved.
    return float(np.clip(sim, 0.0, 1.0))


def blend_and_decide(cv_score, jd_match_score):
    """
    Combine CV-only confidence with JD-aware similarity into a final score,
    then map to a decision based on the configured thresholds.
    """
    final_score = CV_SCORE_WEIGHT * cv_score + JD_MATCH_WEIGHT * jd_match_score

    if final_score >= SHORTLIST_THRESHOLD:
        decision = 'shortlist'
    elif final_score < REJECT_THRESHOLD:
        decision = 'reject'
    else:
        decision = 'review'

    return decision, final_score


# ─── MISTRAL: GENERATE QUESTIONS ───────────────────────────
def generate_questions(candidate, decision, jd_title='', jd_text=''):
    """Generate JD-aware interview questions for shortlist/review candidates."""
    prompt = f"""You are a senior recruiter screening a candidate for a specific job.

JOB POSTING:
Title: {jd_title or 'Not specified'}
Description:
{jd_text or 'Not provided.'}

CANDIDATE:
- Name: {candidate.get('name', '')}
- Skills: {candidate.get('skills', '')}
- Experience: {candidate.get('years_of_experience', '')} years
- Education: {candidate.get('education', '')}
- Screening decision: {decision}

Generate exactly 5 tailored interview questions that probe how well this candidate
fits THIS specific role. Mix technical, behavioral, and gap-probing questions.
Return ONLY a JSON array of 5 strings, no extra text, no markdown."""

    try:
        r = requests.post(
            'https://api.mistral.ai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {MISTRAL_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'mistral-small-latest',
                'messages': [{'role': 'user', 'content': prompt}]
            },
            timeout=30
        )
        data = r.json()

        if 'choices' not in data:
            print(f"   ⚠️  Mistral API returned no 'choices'. Response: {data}")
            return []

        raw = data['choices'][0]['message']['content']
        cleaned = re.sub(r'```json|```', '', raw).strip()
        questions = json.loads(cleaned)

        # Validate: must be a list of strings
        if not isinstance(questions, list):
            print(f"   ⚠️  Mistral did not return a JSON array. Got: {type(questions)}")
            return []

        # Coerce each item to string and drop empties
        questions = [str(q).strip() for q in questions if str(q).strip()]
        return questions[:5]  # cap at 5 in case the model returned more

    except requests.exceptions.Timeout:
        print("   ⚠️  Mistral request timed out")
        return []
    except json.JSONDecodeError as e:
        print(f"   ⚠️  Could not parse Mistral response as JSON: {e}")
        return []
    except Exception as e:
        print(f"   ⚠️  Question generation failed: {e}")
        return []


# ─── POSTGRES: SAVE CANDIDATE ──────────────────────────────
def save_to_postgres(candidate, jd_id, decision, cv_score, jd_match_score,
                     final_score, questions):
    """Insert a candidate row into the Candidate table.

    Note: `cvScore` stores the Spark RF confidence, `jdMatchScore` stores
    the embedding cosine similarity, and `confidence` stores the blended
    final score — which is what the recruiter dashboard surfaces.
    """
    sql = """
        INSERT INTO "Candidate"
          ("jobDescriptionId", "name", "email", "skills", "education",
           "yearsExperience", "cvScore", "jdMatchScore", "finalDecision",
           "confidence", "questions", "createdAt")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING id;
    """
    with pg_conn.cursor() as cur:
        cur.execute(sql, (
            jd_id,
            str(candidate.get('name') or 'Unknown'),
            str(candidate.get('email') or ''),
            str(candidate.get('skills') or ''),
            str(candidate.get('education') or '') or None,
            str(candidate.get('years_of_experience') or '') or None,
            float(cv_score),
            float(jd_match_score),
            str(decision),
            float(final_score),
            format_questions(questions)
        ))
        new_id = cur.fetchone()[0]
        return new_id


# ─── KAFKA SETUP ───────────────────────────────────────────
def safe_json_deserialize(v):
    try:
        return json.loads(v.decode('utf-8'))
    except Exception as e:
        print(f"⚠️  Skipping malformed message: {v[:60]}... ({e})")
        return None


consumer = KafkaConsumer(
    TOPIC_IN,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=safe_json_deserialize,
    auto_offset_reset='earliest',
    group_id='cv-processor-group'
)
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print(f"👂 Listening on topic '{TOPIC_IN}'...")
print("─" * 60)


# ─── MAIN LOOP ─────────────────────────────────────────────
for message in consumer:
    try:
        candidate = message.value
        if candidate is None:
            continue  # skip malformed messages

        sub_id = candidate.get('submission_id', 'no-id')
        print(f"\n📨 New CV received: {candidate.get('name', 'Unknown')} (id: {sub_id})")

        # Read JD context from message
        jd_id = candidate.get('jd_id')
        if jd_id is None:
            print(f"   ⚠️  No jd_id in message — skipping. (Old n8n flow?)")
            error_result = {
                **candidate,
                'decision': 'error',
                'confidence': 0.0,
                'cv_score': 0.0,
                'jd_match_score': 0.0,
                'questions': [],
                'error': 'Missing jd_id — message ignored',
                'processed_at': datetime.now().isoformat()
            }
            producer.send(TOPIC_OUT, error_result)
            producer.flush()
            print("─" * 60)
            continue

        jd_title = candidate.get('jd_title', '?')
        jd_text  = candidate.get('jd_text', '')
        req_skills = candidate.get('required_skills', '')
        print(f"   📋 JD #{jd_id}: \"{jd_title}\"")

        # ── Step 1: Spark ML score (CV-intrinsic) ──────────
        skills_count, years, has_degree = featurize(
            candidate.get('skills'),
            candidate.get('years_of_experience'),
            candidate.get('education')
        )

        df = spark.createDataFrame(
            [(skills_count, years, has_degree)],
            ['skills_count', 'years_clean', 'has_degree']
        )
        assembler = VectorAssembler(
            inputCols=['skills_count', 'years_clean', 'has_degree'],
            outputCol='features'
        )
        df = assembler.transform(df)
        pred = model.transform(df).collect()[0]
        cv_score = float(max(pred.probability))
        print(f"   🎯 CV score (Spark RF):       {cv_score:.3f}")

        # ── Step 2: JD-aware embedding similarity ──────────
        jd_match_score = compute_jd_match_score(candidate, jd_text, req_skills)
        print(f"   🎯 JD match (cosine sim):     {jd_match_score:.3f}")

        # ── Step 3: Blend scores → final decision ──────────
        decision, final_score = blend_and_decide(cv_score, jd_match_score)
        print(f"   🎯 Final blended score:       {final_score:.3f} → {decision.upper()}")

        # ── Step 4: Mistral interview questions (skip rejects) ──
        if decision == 'reject':
            print("   🚫 Rejected — skipping interview questions")
            questions = []
        else:
            print("   💬 Generating JD-aware interview questions...")
            questions = generate_questions(candidate, decision, jd_title, jd_text)
            print(f"   ✅ {len(questions)} questions generated")

        # ── Step 5: Save to Postgres ───────────────────────
        print("   📊 Saving to Postgres...")
        candidate_id = save_to_postgres(
            candidate, jd_id, decision,
            cv_score, jd_match_score, final_score,
            questions
        )
        print(f"   ✅ Saved candidate id: {candidate_id}")

        # ── Step 6: Publish result ─────────────────────────
        result = {
            **candidate,
            'candidate_id': candidate_id,
            'decision': decision,
            'cv_score': cv_score,
            'jd_match_score': jd_match_score,
            'confidence': final_score,  # blended score is what the UI shows
            'questions': questions,
            'processed_at': datetime.now().isoformat()
        }
        producer.send(TOPIC_OUT, result)
        producer.flush()
        print(f"   📤 Published to '{TOPIC_OUT}'")
        print("─" * 60)

    except Exception as e:
        print(f"❌ Error processing message: {e}")
        import traceback
        traceback.print_exc()
        try:
            error_result = {
                **(message.value if message.value else {}),
                'decision': 'error',
                'confidence': 0.0,
                'cv_score': 0.0,
                'jd_match_score': 0.0,
                'questions': [],
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }
            producer.send(TOPIC_OUT, error_result)
            producer.flush()
        except:
            pass
        print("─" * 60)
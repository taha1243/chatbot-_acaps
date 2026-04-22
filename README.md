# Guide Portail ACAPS — Chatbot RAG

> Assistant conversationnel basé sur la **Génération Augmentée par Récupération (RAG)** pour guider les utilisateurs du portail de l'ACAPS (Autorité de Contrôle des Assurances et de la Prévoyance Sociale).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![React](https://img.shields.io/badge/React-18-61DAFB)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![License](https://img.shields.io/badge/Usage-Interne_ACAPS-red)

---

## Sommaire

1. [Présentation](#présentation)
2. [Architecture](#architecture)
3. [Démarrage rapide](#démarrage-rapide)
4. [Services Docker](#services-docker)
5. [Variables d'environnement](#variables-denvironnement)
6. [Pipeline d'ingestion](#pipeline-dingestion)
7. [API REST](#api-rest)
8. [Structure du projet](#structure-du-projet)
9. [Tests](#tests)
10. [Dépannage](#dépannage)

---

## Présentation

Le chatbot guide les utilisateurs du portail ACAPS sur les opérations suivantes :

| Section | Contenu |
|---------|---------|
| **Soumettre une réclamation** | Processus en 6 étapes (Assurance, Prévoyance, Retraite, Mutuelle) |
| **Suivre une réclamation** | Consultation timeline, messages, pièces jointes |
| **Clôture / Réouverture** | Délais, conditions, statuts |
| **Questionnaire de satisfaction** | Évaluation du traitement |

### Fonctionnalités clés

- **RAG ciblé** : recherche uniquement dans `guide_portal_acaps.md`
- **Réponses structurées** : instructions étape par étape avec sources citées
- **LLM local** : Qwen2.5:3b via Ollama — aucune donnée ne quitte le serveur
- **Multilingue** : français / arabe standard / darija marocaine (arabe + translittération latine)
- **Normalisation darija** : mapping automatique darija → français pour améliorer le retrieval
- **Classification d'intention** : détection submit / track / close / satisfaction pour boost contextuel
- **Cache modèle persistant** : volume Docker `atlas-hf-cache`, téléchargement unique (~34 MB)
- **Interface moderne** : design palette officielle ACAPS

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Navigateur  :3000                       │
│              React + Nginx                           │
└────────────────────┬────────────────────────────────┘
                     │  /api/*  (proxy)
                     ▼
┌─────────────────────────────────────────────────────┐
│              Backend API  :8080                      │
│              FastAPI + Uvicorn                       │
│                                                      │
│  [Guardrails] → [Greeting] → [Normalize+Intent]     │
│                                  ↓                   │
│                   [Embedding MiniLM-L12 384-dim]     │
│                                  ↓                   │
│                        [VectorStore pgvector]        │
│                          (guide_portal_acaps.md)     │
│                                  ↓                   │
│                        [LLMClient → Ollama]          │
└──────────────┬──────────────────────┬───────────────┘
               │                      │
               ▼                      ▼
┌─────────────────────┐  ┌──────────────────────────┐
│  PostgreSQL :5432   │  │  Ollama  :11434           │
│  + pgvector         │  │  qwen2.5:3b (1.9 GB)      │
│  atlas_knowledge    │  │  KEEP_ALIVE=-1            │
└─────────────────────┘  └──────────────────────────┘
```

### Flux d'une requête

```
Question utilisateur
      │
      ▼
[InputGuardrails]  ──→ BLOQUÉ si jailbreak / hors-sujet
      │
      ▼
[Greeting?]  ──→ OUI : réponse conversationnelle directe
      │
      ▼
[normalize_query()]  darija → français (DARIJA_MAP 30+ entrées)
      │
      ▼
[classify_intent()]  submit / track / close / satisfaction
      │  → section hint ajouté à la requête de recherche
      ▼
[embed_query()]  MiniLM-L12 → vecteur 384 dimensions
      │
      ▼
[hybrid_search()]  semantic + keyword boost (pg_trgm), TOP_K=3
      │
      ▼
[LLM qwen2.5:3b]  prompt ciblé portail, MAX_TOKENS=512
      │
      ▼
[QueryResponse]  answer + citations + confidence score
```

---

## Démarrage rapide

### Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) ≥ 4.x
- RAM disponible ≥ 3 GB (le modèle Ollama utilise ~2 GB)
- Connexion internet (premier lancement uniquement — téléchargement des modèles)

### Lancement

```bash
# 1. Copier la configuration
cp env.example .env

# 2. Démarrer tous les services
docker compose up -d

# 3. Attendre qu'Ollama télécharge et charge qwen2.5:3b (~5 min)
docker logs atlas-ollama -f
#    ✓  Modèle chargé en RAM

# 4. Attendre que l'API pre-warm le modèle d'embedding (~1 min)
docker logs atlas-api -f
#    ✓  Embedding model pre-loaded successfully

# 5. Indexer le guide portail
docker exec atlas-api python -m data_ingestion.pipeline --recreate

# 6. Ouvrir l'interface
# http://localhost:3000
```

> **Note :** Le premier démarrage télécharge qwen2.5:3b (1.9 GB) et le modèle d'embedding (~34 MB).  
> Les démarrages suivants sont instantanés — tout est mis en cache dans les volumes Docker (`ollama_data`, `atlas-hf-cache`).

> **Important :** Lancer l'ingestion (étape 5) uniquement après que l'API affiche "pre-loaded successfully".  
> Lancer les deux simultanément provoquerait un conflit de téléchargement du modèle d'embedding.

---

## Services Docker

| Service | Image | Port | Rôle |
|---------|-------|------|------|
| `postgres` | pgvector/pgvector:pg16 | 5432 | Base vectorielle |
| `pgadmin` | dpage/pgadmin4:8.10 | 5050 | Administration BDD |
| `ollama` | build: ./ollama | 11434 | LLM local (qwen2.5:3b) |
| `api` | build: ./backend_api | 8080 | Backend FastAPI |
| `ui` | build: ./frontend_ui | 3000 | Interface React + Nginx |

**Accès :**

| Interface | URL | Identifiants |
|-----------|-----|-------------|
| Chatbot | http://localhost:3000 | — |
| API docs (Swagger) | http://localhost:8080/docs | — |
| pgAdmin | http://localhost:5050 | admin@atlas.com / admin |
| Santé API | http://localhost:8080/health | — |

---

## Variables d'environnement

Fichier `.env` (copie de `env.example`) :

```bash
# LLM
VLLM_MODEL=qwen2.5:3b
VLLM_URL=http://ollama:11434/v1
VLLM_API_KEY=ollama
LLM_PROVIDER=ollama

# Base de données
DATABASE_URL=postgresql://atlas:atlas@postgres:5432/atlas_rag
POSTGRES_DB=atlas_rag
POSTGRES_USER=atlas
POSTGRES_PASSWORD=atlas

# Embedding (AutoTokenizer + AutoModel via transformers)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSION=384

# RAG
SIMILARITY_THRESHOLD=0.40
TOP_K_RESULTS=3
MAX_TOKENS=512
TEMPERATURE=0.0

# Sécurité
ENABLE_GUARDRAILS=true
```

**Changer de LLM provider :**

| Provider | VLLM_URL | LLM_PROVIDER |
|----------|----------|--------------|
| Ollama (défaut) | `http://ollama:11434/v1` | `ollama` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openrouter` |
| OpenAI | `https://api.openai.com/v1` | `openai` |

---

## Pipeline d'ingestion

Les documents sources sont des fichiers Markdown dans `data_ingestion/documents/`.

**Document actif :**
- `guide_portal_acaps.md` — Guide officiel d'utilisation du portail (17 chunks)

### Format attendu

```markdown
---
title: Guide d'Utilisation du Portail ACAPS
version: 1.0
---

# Section 1 – Titre

## Étape 1
- Instruction A
- Instruction B
```

### Commandes d'ingestion

```bash
# Réindexer complètement (recommandé après modification du guide ou changement de dimension)
docker exec atlas-api python -m data_ingestion.pipeline --recreate

# Ingestion incrémentale (seulement les fichiers modifiés)
docker exec atlas-api python -m data_ingestion.pipeline
```

---

## API REST

### `POST /chat`

```json
// Requête
{
  "question": "Comment soumettre une réclamation ?",
  "conversation_id": "uuid-optionnel"
}

// Réponse
{
  "answer": "Pour soumettre une réclamation, suivez les étapes suivantes...",
  "citations": [
    {
      "title": "Section 1 – Soumettre une Réclamation > Étape 1",
      "url": "",
      "score": 0.57,
      "snippet": "Saisissez votre nom et prénom..."
    }
  ],
  "confidence": 0.57,
  "conversation_id": "uuid",
  "metadata": { "retrieval_count": 3, "model": "qwen2.5:3b", "intent": "submit" }
}
```

### Autres endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Santé des composants |
| `GET` | `/stats` | Statistiques vector store |
| `POST` | `/ingest` | Déclencher ingestion |
| `POST` | `/feedback` | Feedback utilisateur |

---

## Structure du projet

```
chatbot-_acaps/
├── backend_api/
│   ├── app/
│   │   ├── main.py          # Endpoints FastAPI, lifespan + embedding pre-warm
│   │   ├── engine.py        # Moteur RAG (normalize → intent → embed → search → LLM)
│   │   ├── models.py        # Schémas Pydantic
│   │   ├── config.py        # Settings (pydantic-settings)
│   │   └── guardrails/      # Validation input/output
│   ├── Dockerfile           # Multi-stage, torch CPU-only, run as root
│   └── requirements.txt
│
├── data_ingestion/
│   ├── documents/
│   │   └── guide_portal_acaps.md   # Source unique (17 chunks)
│   ├── parser.py            # Markdown → DocumentChunk[]
│   ├── embedder.py          # AutoTokenizer+AutoModel MiniLM-L12 (384-dim)
│   ├── vector_store.py      # PostgreSQL + pgvector, hybrid_search()
│   └── pipeline.py          # Orchestration ingestion
│
├── frontend_ui/
│   ├── src/
│   │   ├── App.tsx          # Interface chat React, palette ACAPS
│   │   └── index.css        # Styles ACAPS (sky/navy/sand)
│   ├── nginx.conf           # Proxy /api, timeout 300s
│   └── Dockerfile           # Node build → Nginx serve
│
├── ollama/
│   ├── Dockerfile           # ollama/ollama + python3
│   ├── entrypoint.sh        # Pull + warm-up modèle
│   └── progress_parser.py   # Affichage progression téléchargement
│
├── tests/
│   ├── test_pipeline_full.py   # 62 questions (FR / AR / Darija / mix)
│   └── test_results.json       # Résultats dernière exécution
│
├── docker-compose.yml
├── env.example
├── commandes.txt            # Référence commandes Docker
└── README.md
```

---

## Tests

Le script `tests/test_pipeline_full.py` valide le pipeline complet sur 62 questions réparties en 12 catégories.

```bash
# Prérequis : API healthy + ingestion terminée
PYTHONIOENCODING=utf-8 python tests/test_pipeline_full.py
```

### Résultats (dernière exécution — 2026-04-22)

| Catégorie | ✅ | 🔄 | Note |
|-----------|----|----|------|
| Basiques FR | 6/7 | 1 | "email obligatoire" — phrasing |
| Basiques AR | 7/7 | 0 | Parfait |
| Reformulation FR | 4/6 | 2 | "me plaindre" + "état dossier" |
| Reformulation AR/Darija | 5/6 | 1 | Darija pur sans script arabe |
| Pièges FR | 4/5 | 1 | "téléphone obligatoire" phrasing |
| Pièges AR | 4/5 | 1 | Taille fichier non couverte |
| Scénario FR | 4/5 | 1 | "référence perdue" — contenu absent |
| Scénario AR | 5/5 | 0 | Parfait |
| Multi-étapes FR | 3/4 | 1 | assurance/prévoyance — contenu absent |
| Multi-étapes AR | 3/4 | 1 | idem |
| Bruit FR+Darija | 2/4 | 2 | Translittération latine non supportée |
| Bruit AR mix | 4/4 | 0 | Parfait |
| **TOTAL** | **51/62** | **11/62** | **0 bloqués, 1 erreur réseau** |

> 🔄 = réponse fournie via LLM general knowledge, sans citation documentaire  
> Les améliorations prioritaires : synonymes français informels + contenu manquant dans le guide

---

## Dépannage

### Le chatbot répond "erreur survenue"

**Cause :** Ollama est encore en train de charger le modèle (première requête après démarrage).

```bash
docker logs atlas-ollama --tail=5
# Attendre que qwen2.5:3b soit chargé
```

### Réponse "je ne peux pas trouver cette information"

**Cause :** La base vectorielle est vide ou le modèle d'embedding n'est pas encore chargé.

```bash
# Vérifier que l'API a fini de charger le modèle d'embedding
docker logs atlas-api --tail=10
# Chercher : "Embedding model pre-loaded successfully"

# Puis lancer l'ingestion
docker exec atlas-api python -m data_ingestion.pipeline --recreate
```

### Conteneur api redémarre en boucle (exit code 137)

**Cause :** Manque de RAM (OOM kill). MiniLM + Ollama nécessitent ~3 GB RAM disponible.

**Solution :** Fermer les applications consommant de la RAM. Vérifier dans Docker Desktop → Resources → Memory ≥ 4 GB.

### Conflit téléchargement modèle d'embedding

**Cause :** L'API et la commande d'ingestion téléchargent le modèle simultanément au premier lancement.

**Solution :** Attendre que l'API affiche "pre-loaded successfully" avant de lancer l'ingestion.

### Vérifier l'état général

```bash
docker compose ps
curl http://localhost:8080/health
curl http://localhost:8080/stats
```

---

## Licence

Usage interne ACAPS — © 2026 ACAPS. Tous droits réservés.

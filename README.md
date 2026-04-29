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
4. [Intégration dans une application existante](#intégration-dans-une-application-existante)
5. [Référence API](#référence-api)
6. [Variables d'environnement](#variables-denvironnement)
7. [Configuration métier (JSON)](#configuration-métier-json)
8. [Pipeline d'ingestion](#pipeline-dingestion)
9. [Services Docker](#services-docker)
10. [Structure du projet](#structure-du-projet)
11. [Tests](#tests)
12. [Dépannage](#dépannage)

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

- **RAG strictement ancré** : recherche dans le guide officiel uniquement (FR + AR)
- **Streaming SSE** : réponses token-par-token via `/chat/stream`
- **LLM local** : Qwen2.5 via Ollama — aucune donnée ne quitte le serveur
- **Multilingue** : français / arabe standard, deux guides séparés (`guide_portal_acaps 3.md` / `guide_portal_acaps_ar.md`)
- **Guardrails** : input (jailbreak, off-topic, toxicité) et output (ancrage dans le contexte)
- **Vérification d'ancrage** : la réponse doit partager au moins 50 % de ses mots-clés avec le contexte récupéré, sinon elle est remplacée par un message « hors guide »
- **Cache modèle persistant** : volumes Docker `atlas-hf-cache` et `atlas-ollama-data`
- **Interface** : React + Vite + Tailwind, palette officielle ACAPS

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Navigateur  :3000                       │
│              React + Nginx                           │
└────────────────────┬────────────────────────────────┘
                     │  /api/*  (proxy Nginx)
                     ▼
┌─────────────────────────────────────────────────────┐
│              Backend API  :8080                      │
│              FastAPI + Uvicorn                       │
│                                                      │
│  [InputGuardrails] → [Greeting?] → [Lang detect]    │
│                                       ↓              │
│                    [Embedding multilingual-e5-base]  │
│                       768-dim                        │
│                                       ↓              │
│                    [VectorStore pgvector hybrid]     │
│                       (atlas_knowledge)              │
│                                       ↓              │
│                    [Ollama LLM]  →  [Output guard]   │
│                                     [Grounding chk]  │
└──────────────┬──────────────────────┬───────────────┘
               │                      │
               ▼                      ▼
┌─────────────────────┐  ┌──────────────────────────┐
│  PostgreSQL :5432   │  │  Ollama  :11434           │
│  + pgvector         │  │  qwen2.5:3b (1.9 GB)      │
│  atlas_knowledge    │  │                            │
└─────────────────────┘  └──────────────────────────┘
```

### Flux d'une requête

```
Question utilisateur
      │
      ▼
[InputGuardrails]      → BLOQUÉ si jailbreak / off-topic / toxique
      │
      ▼
[Greeting?]            → OUI : réponse conversationnelle directe
      │
      ▼
[detect_language]      → FR ou AR (charge le guide correspondant)
      │
      ▼
[embed_query()]        → vecteur 768-dim (multilingual-e5-base)
      │
      ▼
[hybrid_search()]      → semantic + keyword, top_k filtré sur le guide
      │
      ▼
[grounding check]      → si avg_score < 0.40 : retour "hors guide"
      │
      ▼
[LLM qwen2.5:3b]       → génère la réponse à partir du contexte
      │
      ▼
[post_validate]        → vérifie ancrage de la réponse (≥ 50 % tokens
      │                   présents dans le contexte) sinon "hors guide"
      ▼
QueryResponse / SSE stream
```

---

## Démarrage rapide

### Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) ≥ 4.x
- RAM disponible ≥ 4 GB (Ollama ~2 GB + embedding ~1 GB)
- Connexion internet (premier lancement uniquement — téléchargement des modèles)

### Lancement

```bash
# 1. Configuration
cp env.example .env

# 2. Démarrer tous les services
docker compose up -d

# 3. Suivre le téléchargement de qwen2.5:3b (~5 min, une seule fois)
docker logs atlas-ollama -f

# 4. Suivre le pre-warm de l'embedding model (~1 min, une seule fois)
docker logs atlas-api -f
#    Attendre: "Embedding model pre-loaded successfully"

# 5. Indexer les guides (FR + AR)
docker exec atlas-api python -m data_ingestion.pipeline --recreate

# 6. Ouvrir l'interface
# http://localhost:3000
```

> **Premier démarrage** : qwen2.5:3b (1.9 GB) + multilingual-e5-base (~1.1 GB) sont téléchargés et mis en cache dans les volumes `atlas-ollama-data` et `atlas-hf-cache`. Les redémarrages suivants prennent < 30 s.

> **Important** : ne pas lancer l'ingestion (étape 5) tant que l'API n'a pas affiché `Embedding model pre-loaded successfully` — sinon conflit de téléchargement.

---

## Intégration dans une application existante

Trois stratégies au choix selon le besoin.

### Option A — Embarquer l'UI complète via `iframe`

La plus simple : on déploie la stack telle quelle et on embarque le frontend dans l'application hôte.

```html
<iframe
  src="https://chatbot.acaps.local/"
  style="border:0; width:420px; height:640px;"
  title="Assistant ACAPS"
  allow="clipboard-write"
></iframe>
```

Configuration côté serveur :

- Mettre `frontend_ui` derrière le même reverse proxy que l'application hôte (ou un sous-domaine dédié).
- Si l'application hôte et le chatbot sont sur des origines différentes, ajuster `CORS_ORIGINS` (voir [Variables d'environnement](#variables-denvironnement)) et autoriser l'origine hôte dans la CSP de l'iframe.

### Option B — Appel direct à l'API (REST classique)

Pour une intégration native (composant React/Vue/Angular dans l'application existante) qui veut sa propre UI.

```ts
// Exemple TypeScript
async function askAcaps(question: string, conversationId?: string) {
  const res = await fetch('https://chatbot.acaps.local/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  });
  if (!res.ok) throw new Error('chatbot_unavailable');
  return res.json(); // { answer, citations[], confidence, conversation_id, metadata }
}
```

### Option C — Streaming Server-Sent Events (recommandé UX)

Affichage token-par-token, identique à ce que fait l'UI fournie.

```ts
async function askAcapsStream(
  question: string,
  conversationId: string | undefined,
  onToken: (text: string, replace?: boolean) => void,
  onMeta: (meta: { citations: Citation[]; confidence: number }) => void,
  onDone: () => void,
) {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  });
  if (!res.ok || !res.body) throw new Error('network_error');

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = 'message';
      const dataLines: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
      }
      const data = dataLines.length ? JSON.parse(dataLines.join('\n')) : {};
      if (event === 'meta')  onMeta(data);
      if (event === 'token') onToken(data.text ?? '', !!data.replace);
      if (event === 'done')  onDone();
    }
  }
}
```

**Événements SSE émis :**

| Événement | Payload | Quand |
|-----------|---------|-------|
| `start`   | `{ conversation_id }` | Une fois, à l'ouverture du flux |
| `meta`    | `{ citations, confidence, metadata }` | Une fois, après la recherche |
| `token`   | `{ text, replace? }` | N fois, pour chaque chunk de texte. `replace: true` indique que la réponse doit être réécrite intégralement (post-validation a échoué) |
| `done`    | `{}` | Une fois, fin du flux |
| `error`   | `{ message }` | En cas d'erreur fatale |

> Si le proxy de l'application hôte est Nginx, **désactiver le buffering** sur cette route :
> ```nginx
> location /api/chat/stream {
>     proxy_pass http://api:8080/chat/stream;
>     proxy_buffering off;
>     proxy_read_timeout 300s;
>     proxy_set_header X-Accel-Buffering no;
> }
> ```

### Configuration CORS

Par défaut `CORS_ORIGINS=*`. Pour la production, restreindre :

```bash
# .env
CORS_ORIGINS=https://portal.acaps.ma,https://intranet.acaps.ma
```

### Authentification

Le backend ne gère **pas** l'authentification — il est conçu pour être déployé derrière un reverse-proxy ou un API gateway qui fait l'auth (mTLS, JWT, OAuth2…). À ajouter au niveau Nginx/Traefik en amont.

Si une auth applicative est nécessaire, ajouter une dépendance FastAPI dans [backend_api/app/main.py](backend_api/app/main.py) sur les routes `/chat` et `/chat/stream`.

### Persistance de la conversation

Le `conversation_id` est généré côté serveur si non fourni. **Aucun historique n'est stocké côté backend** — l'application cliente est responsable de :

1. Conserver le `conversation_id` retourné dans la première réponse (`event: start` en SSE, ou champ `conversation_id` en REST).
2. Le renvoyer dans les requêtes suivantes pour permettre une corrélation des logs côté serveur.

> Important : actuellement le backend ne ré-injecte pas l'historique dans le prompt. Si vous avez besoin de mémoire conversationnelle, il faut l'implémenter côté client (passer l'historique dans `question`) ou étendre [engine.py](backend_api/app/engine.py) (`RAGEngine.query`).

---

## Référence API

Documentation Swagger interactive : `http://localhost:8080/docs`

### `POST /chat` — Réponse synchrone

**Requête**
```json
{
  "question": "Comment soumettre une réclamation ?",
  "conversation_id": "uuid-optionnel",
  "language": "auto"
}
```

**Réponse**
```json
{
  "answer": "Pour soumettre une réclamation, suivez les étapes suivantes…",
  "citations": [
    {
      "title": "Section 1 – Soumettre une Réclamation > Étape 1",
      "url": "",
      "score": 0.57,
      "snippet": "Saisissez votre nom et prénom..."
    }
  ],
  "confidence": 0.57,
  "conversation_id": "f3a9…",
  "metadata": {
    "retrieval_count": 3,
    "top_score": 0.62,
    "model": "qwen2.5:3b"
  }
}
```

### `POST /chat/stream` — Server-Sent Events

Même payload que `/chat`. Réponse en `text/event-stream` (voir [Option C](#option-c--streaming-server-sent-events-recommandé-ux)).

### `GET /health`

```json
{
  "status": "healthy",
  "components": { "vector_store": true, "llm": true, "embedder": true, "overall": true },
  "version": "1.0.0",
  "timestamp": "2026-04-28T08:30:00Z"
}
```

### `GET /stats`

```json
{
  "vector_store": { "total_chunks": 17, "table": "atlas_knowledge" },
  "config": {
    "model": "qwen2.5:3b",
    "embedding_model": "intfloat/multilingual-e5-base",
    "top_k": 8,
    "threshold": 0.25
  }
}
```

### `POST /ingest`

Réindexer un fichier ou un dossier sans redémarrer l'API.

```json
{
  "file_path": "data_ingestion/documents/guide_portal_acaps 3.md",
  "directory": null,
  "recreate_collection": false
}
```

### `POST /feedback`

```json
{ "conversation_id": "f3a9…", "rating": 4, "helpful": true, "comment": "Réponse claire" }
```

### Codes d'erreur

| Statut | Cause | Comportement client recommandé |
|--------|-------|-------------------------------|
| `400`  | Validation Pydantic (`question` vide ou > 2000 chars) | Corriger l'input |
| `500`  | Erreur LLM / vector store | Retry après quelques secondes |
| `503`  | API en cours de pre-warm | Attendre le `/health` `overall: true` |

---

## Variables d'environnement

Fichier `.env` à la racine (copie de `env.example`).

> ⚠️ **Note importante** : le backend lit les variables `LLM_*` (et non `VLLM_*` qui apparaissent dans `env.example` pour des raisons historiques). Les variables effectivement utilisées sont celles listées ci-dessous, qui correspondent à `docker-compose.yml`.

```bash
# LLM (Ollama OpenAI-compatible)
LLM_MODEL=qwen2.5:3b
LLM_BASE_URL=http://ollama:11434/v1
LLM_API_KEY=ollama

# Base de données
DATABASE_URL=postgresql://atlas:atlas@postgres:5432/atlas_rag
VECTOR_TABLE=atlas_knowledge
POSTGRES_DB=atlas_rag
POSTGRES_USER=atlas
POSTGRES_PASSWORD=atlas

# Embedding
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DIMENSION=768

# RAG
SIMILARITY_THRESHOLD=0.25      # seuil de filtrage par chunk (vector store)
TOP_K_RESULTS=8
TEMPERATURE=0
MAX_TOKENS=600

# Sécurité
ENABLE_GUARDRAILS=true
LOG_LEVEL=INFO
DEBUG=false
CORS_ORIGINS=*
```

### Changer de provider LLM

Le backend utilise une API OpenAI-compatible. Tout provider compatible fonctionne :

| Provider | `LLM_BASE_URL` | `LLM_API_KEY` | `LLM_MODEL` |
|----------|----------------|---------------|-------------|
| Ollama (défaut) | `http://ollama:11434/v1` | `ollama` | `qwen2.5:3b` |
| OpenRouter | `https://openrouter.ai/api/v1` | `sk-or-…` | `qwen/qwen-2.5-7b-instruct` |
| OpenAI | `https://api.openai.com/v1` | `sk-…` | `gpt-4o-mini` |

---

## Configuration métier (JSON)

Trois fichiers permettent d'ajuster le comportement **sans rebuild** :

### [backend_api/config/engine_config.json](backend_api/config/engine_config.json)

```json
{
  "portal_file_fr": "guide_portal_acaps 3.md",
  "portal_file_ar": "guide_portal_acaps_ar.md",
  "max_chunk_chars_for_prompt": 500,
  "min_confidence_threshold": 0.40,
  "high_confidence_skip_hallucination": 0.50,
  "min_answer_length": 5,
  "answer_grounding_min_ratio": 0.5,
  "answer_grounding_min_tokens": 3,
  "greeting_keywords": ["bonjour", "salut", ...]
}
```

| Clé | Effet |
|-----|-------|
| `min_confidence_threshold` | Score moyen minimum pour considérer la requête « ancrée » dans le guide. Si en-dessous → message « hors guide ». |
| `answer_grounding_min_ratio` | Proportion minimale de mots-clés de la réponse qui doivent apparaître dans le contexte récupéré. Évite que le LLM réponde de mémoire (ex. capitales, dates, etc.). |
| `answer_grounding_min_tokens` | Réponses plus courtes que ce nombre de tokens « content » sautent la vérification d'ancrage. |

### [backend_api/config/prompts.json](backend_api/config/prompts.json)

Prompts système et utilisateur en français et en arabe. Modifier ici pour :
- changer le ton de l'assistant
- restreindre/élargir le périmètre
- adapter à un autre domaine fonctionnel

### [backend_api/config/guardrails_config.json](backend_api/config/guardrails_config.json)

Patterns regex pour la détection d'attaques d'injection, de toxicité, de hors-sujet, et messages de blocage en FR/AR.

> Les fichiers de config sont montés en volume read-only (`./backend_api/config:/app/config:ro`). Pour les recharger : `docker compose restart api`.

---

## Pipeline d'ingestion

Les documents sources sont des fichiers Markdown dans `data_ingestion/documents/`.

**Documents actifs :**
- `guide_portal_acaps 3.md` — guide officiel français
- `guide_portal_acaps_ar.md` — version arabe

### Format Markdown attendu

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

### Commandes

```bash
# Réindexation complète (recommandé après modification du guide)
docker exec atlas-api python -m data_ingestion.pipeline --recreate

# Incrémentale (seuls les fichiers modifiés)
docker exec atlas-api python -m data_ingestion.pipeline

# Via l'API (sans entrer dans le container)
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{"recreate_collection": true}'
```

### Ajouter un nouveau document

1. Déposer le fichier `.md` dans `data_ingestion/documents/`
2. Si la langue diffère, mettre à jour `engine_config.json` (`portal_file_fr` ou `portal_file_ar`)
3. Lancer `--recreate`
4. Vérifier `GET /stats` pour confirmer le nouveau nombre de chunks

---

## Services Docker

| Service | Image | Port | Rôle |
|---------|-------|------|------|
| `postgres` | pgvector/pgvector:pg16 | 5432 | Base vectorielle |
| `pgadmin` | dpage/pgadmin4:8.10 | 5050 | Administration BDD |
| `ollama` | ollama/ollama:latest | 11434 | LLM local (qwen2.5:3b) |
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

## Structure du projet

```
ChatAssistantAcaps/
├── backend_api/
│   ├── app/
│   │   ├── main.py          # Endpoints FastAPI (chat, stream, health, ingest, stats)
│   │   ├── engine.py        # Moteur RAG (lang detect → embed → search → LLM → grounding)
│   │   ├── models.py        # Schémas Pydantic
│   │   ├── config.py        # Settings (pydantic-settings)
│   │   └── guardrails/      # Validation input/output (regex)
│   ├── config/
│   │   ├── engine_config.json     # Seuils RAG, fichiers source
│   │   ├── prompts.json           # Prompts FR / AR
│   │   └── guardrails_config.json # Patterns + messages bloqués
│   ├── Dockerfile
│   └── requirements.txt
│
├── data_ingestion/
│   ├── documents/
│   │   ├── guide_portal_acaps 3.md   # Source FR
│   │   └── guide_portal_acaps_ar.md  # Source AR
│   ├── parser.py            # Markdown → DocumentChunk[]
│   ├── embedder.py          # multilingual-e5-base (768-dim)
│   ├── vector_store.py      # PostgreSQL + pgvector + hybrid_search()
│   └── pipeline.py          # Orchestration ingestion
│
├── frontend_ui/
│   ├── src/
│   │   ├── App.tsx          # Interface chat (SSE streaming)
│   │   ├── main.tsx         # Bootstrap React
│   │   └── index.css        # Tailwind + thème ACAPS
│   ├── nginx.conf           # Proxy /api, SSE buffering off
│   ├── vite.config.ts
│   └── Dockerfile           # Vite build → Nginx serve
│
├── ollama/
│   └── Dockerfile
│
├── tests/
│   ├── test_check.py
│   └── testfinale.py        # Banque de questions FR/AR/Darija
│
├── docker-compose.yml
├── env.example
├── commandes.txt            # Référence commandes Docker
└── README.md
```

---

## Tests

Banque de questions dans [tests/testfinale.py](tests/testfinale.py) couvrant FR, AR, darija et cas piégeux (questions hors guide, jailbreaks, etc.).

```bash
# Prérequis : API healthy + ingestion terminée
PYTHONIOENCODING=utf-8 python tests/testfinale.py
```

Pour un smoke-test rapide :

```bash
curl -s http://localhost:8080/health | jq .
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Comment soumettre une réclamation ?"}' | jq .
```

---

## Dépannage

### Le chatbot répond « erreur survenue »
Ollama est encore en train de charger le modèle (première requête après démarrage).
```bash
docker logs atlas-ollama --tail=20
```

### Réponse « Je ne trouve pas cette information dans les documents… »
Trois causes possibles :
1. La base vectorielle est vide → lancer l'ingestion (`docker exec atlas-api python -m data_ingestion.pipeline --recreate`).
2. Le score moyen de retrieval est < `min_confidence_threshold` (0.40) → soit la question est légitimement hors guide, soit ajuster le seuil dans `engine_config.json`.
3. La vérification d'ancrage a échoué → consulter `docker logs atlas-api` pour le message `Answer not grounded — overlap …`.

### Conteneur `api` redémarre en boucle (exit code 137)
OOM kill. Augmenter la RAM allouée à Docker Desktop (≥ 4 GB) ou réduire `MAX_TOKENS`.

### Conflit de téléchargement du modèle d'embedding
Lancer l'ingestion uniquement après `Embedding model pre-loaded successfully` dans `docker logs atlas-api`.

### CORS bloque les requêtes depuis l'application hôte
Mettre l'origine de l'hôte dans `CORS_ORIGINS` puis `docker compose restart api`.

### Le streaming SSE coupe au bout de quelques tokens
Vérifier que le proxy en amont **désactive le buffering** sur `/chat/stream` (cf. [Option C](#option-c--streaming-server-sent-events-recommandé-ux)).

### État général en une commande
```bash
docker compose ps
curl http://localhost:8080/health
curl http://localhost:8080/stats
```

---

## Licence

Usage interne ACAPS — © 2026 ACAPS. Tous droits réservés.

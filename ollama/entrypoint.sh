#!/bin/bash
set -e

MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"

echo ">>> Démarrage du serveur Ollama..."
ollama serve &
OLLAMA_PID=$!

echo ">>> Attente que l'API soit prête..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo ">>> Ollama est prêt."

if ollama list | grep -q "${MODEL}"; then
    echo ">>> Modèle ${MODEL} déjà présent, pas de téléchargement."
else
    echo ">>> Téléchargement du modèle ${MODEL}..."
    curl -sN -X POST http://localhost:11434/api/pull \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${MODEL}\"}" | python3 -u /progress_parser.py
    echo ">>> Modèle ${MODEL} téléchargé avec succès."
fi

echo ">>> Chargement du modèle en mémoire (warm-up)..."
curl -s -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"prompt\": \"Bonjour\", \"stream\": false}" \
    --max-time 120 > /dev/null 2>&1 && echo ">>> Modèle chargé en RAM." || echo ">>> Warm-up timeout, modèle sera chargé à la première requête."

echo ">>> Serveur Ollama opérationnel avec ${MODEL} (keep_alive=-1)."
wait $OLLAMA_PID

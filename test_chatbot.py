"""
Test script — 40 questions sur le chatbot ACAPS
Usage: python test_chatbot.py
"""
import json
import urllib.request
import urllib.error
import time

API_URL = "http://localhost:8080/chat"

QUESTIONS = [
    # Section 1 — FR
    (1,  "Comment soumettre une réclamation sur le portail ACAPS ?"),
    (2,  "Quelles sont les étapes pour déposer une réclamation ?"),
    (3,  "Quels documents faut-il joindre à la réclamation ?"),
    (4,  "Est-ce que le numéro de téléphone est obligatoire ?"),
    (5,  "Est-ce que l'email est obligatoire pour créer une réclamation ?"),
    (6,  "Quelle est la taille maximale des fichiers à joindre ?"),
    (7,  "Quels formats de fichiers sont acceptés ?"),
    (8,  "Je veux me plaindre contre ma compagnie d'assurance, comment faire ?"),
    (9,  "Comment remplir le formulaire de réclamation ?"),
    (10, "Quelles informations dois-je saisir dans le formulaire ?"),
    # Section 1 — AR
    (11, "كيف أقدم شكاية على بوابة ACAPS ؟"),
    (12, "ما هي الوثائق المطلوبة لتقديم الشكاية ؟"),
    (13, "هل البريد الإلكتروني إلزامي ؟"),
    (14, "ما هي خطوات تقديم شكاية ؟"),
    (15, "ما هو الحجم الأقصى للملفات المرفقة ؟"),
    # Section 2 — FR
    (16, "Comment suivre l'état de ma réclamation ?"),
    (17, "Comment consulter ma réclamation après envoi ?"),
    (18, "J'ai perdu mon numéro de référence, que faire ?"),
    (19, "Combien de temps faut-il pour avoir une réponse ?"),
    (20, "Comment ajouter un document après avoir soumis ma réclamation ?"),
    (21, "Je n'ai pas reçu de réponse, que dois-je faire ?"),
    # Section 2 — AR
    (22, "كيف أتابع شكايتي ؟"),
    (23, "ضيعت رقم المرجع، ماذا أفعل ؟"),
    (24, "كيف أعرف حالة شكايتي ؟"),
    # Section 3 — FR
    (25, "Comment clôturer ma réclamation ?"),
    (26, "Comment réouvrir une réclamation fermée ?"),
    (27, "Dans quel délai ma réclamation est-elle clôturée automatiquement ?"),
    (28, "Que se passe-t-il si je ne réponds pas dans les 7 jours ?"),
    (29, "Est-ce que ma réclamation reste ouverte indéfiniment ?"),
    # Section 3 — AR
    (30, "كيف أغلق شكايتي ؟"),
    (31, "كيف أعيد فتح شكاية مغلقة ؟"),
    (32, "ما هو الأجل قبل الإغلاق التلقائي ؟"),
    # Section 4 — FR
    (33, "Comment remplir le questionnaire de satisfaction ?"),
    (34, "Je n'ai pas reçu le questionnaire de satisfaction, pourquoi ?"),
    (35, "À quel moment reçoit-on le questionnaire de satisfaction ?"),
    # Section 4 — AR
    (36, "لم أتلق استبيان الرضا، ما السبب ؟"),
    # Salutations
    (37, "Bonjour"),
    (38, "مرحبا"),
    # Hors sujet
    (39, "Quel est le cours du dollar aujourd'hui ?"),
    (40, "Explique-moi comment fonctionne l'intelligence artificielle."),
]

WIDTH_Q = 55

def ask(question: str) -> dict:
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))

def truncate(text: str, n: int) -> str:
    text = text.replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text

def run():
    print("=" * 110)
    print(f"{'#':<4} {'Question':<{WIDTH_Q}} {'Conf':>5}  {'Citations':<35}  Réponse")
    print("=" * 110)

    results = {"ok": 0, "warn": 0, "err": 0}

    for num, question in QUESTIONS:
        try:
            r = ask(question)
            confidence = r.get("confidence", 0.0)
            answer = r.get("answer", "")
            citations = r.get("citations", [])
            metadata = r.get("metadata", {})

            source = metadata.get("source", "")
            greeting = metadata.get("greeting", False)

            # Determine status
            if greeting:
                status = "✓ GREETING"
                results["ok"] += 1
            elif source == "general_redirect":
                status = "↪ REDIRECT"
                results["ok"] += 1
            elif confidence >= 0.25 and answer and "n'est pas dans le guide" not in answer.lower():
                status = "✓ OK"
                results["ok"] += 1
            elif "n'est pas dans le guide" in answer.lower() or "ليس في الدليل" in answer or not answer:
                status = "✗ NO ANS"
                results["warn"] += 1
            else:
                status = "~ LOW"
                results["warn"] += 1

            cit_str = truncate(", ".join(c["title"].split(">")[-1].strip() for c in citations[:2]), 35)
            ans_str = truncate(answer, 60)
            q_str   = truncate(question, WIDTH_Q)

            print(f"{num:<4} {q_str:<{WIDTH_Q}} {confidence:>5.2f}  {cit_str:<35}  [{status}] {ans_str}")

        except Exception as e:
            q_str = truncate(question, WIDTH_Q)
            print(f"{num:<4} {q_str:<{WIDTH_Q}}  ERR    {'':35}  [✗ ERROR] {e}")
            results["err"] += 1

        time.sleep(0.3)

    print("=" * 110)
    total = len(QUESTIONS)
    print(f"RÉSULTAT : ✓ {results['ok']}/{total}   ~ {results['warn']} avertissements   ✗ {results['err']} erreurs")
    print("=" * 110)

if __name__ == "__main__":
    run()

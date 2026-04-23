"""
Full pipeline test — all question categories.
Run after ingestion: python tests/test_pipeline_full.py
"""
import json
import time
import sys
import urllib.request
import urllib.error

API = "http://localhost:8080/chat"
TIMEOUT = 180  # seconds per question (CPU LLM is slow)

CATEGORIES = {
    "🇫🇷 Basiques FR": [
        "Comment soumettre une réclamation ?",
        "Quelles sont les étapes pour déposer une réclamation ?",
        "Quels documents peut-on joindre à une réclamation ?",
        "Est-ce que l'email est obligatoire ?",
        "Comment suivre une réclamation ?",
        "Que faire après avoir reçu une réponse ?",
        "Combien de temps pour réouvrir une réclamation ?",
    ],
    "🇸🇦 Basiques AR": [
        "كيف يمكنني تقديم شكاية؟",
        "ما هي خطوات تقديم الشكاية؟",
        "ما هي الوثائق التي يمكن إرفاقها؟",
        "هل البريد الإلكتروني إجباري؟",
        "كيف يمكنني تتبع الشكاية؟",
        "ماذا أفعل بعد تلقي الرد؟",
        "كم المدة المتاحة لإعادة فتح الشكاية؟",
    ],
    "🎯 Reformulation FR": [
        "Je veux me plaindre, je fais comment ?",
        "Où est le bouton pour déposer une plainte ?",
        "Je peux envoyer une réclamation sans téléphone ?",
        "Comment voir l'état de mon dossier ?",
        "J'ai déjà envoyé une réclamation, comment la consulter ?",
        "Est-ce que je peux modifier ma réclamation après envoi ?",
    ],
    "🎯 Reformulation AR (Darija)": [
        "بغيت ندير شكاية، كيفاش ندير؟",
        "فين نلقى زر تقديم الشكاية؟",
        "واش نقدر ندير شكاية بلا رقم الهاتف؟",
        "كيف أطلع على حالة الشكاية ديالي؟",
        "رسلت شكاية، كيف نشوف التفاصيل ديالها؟",
        "واش يمكن نبدل الشكاية من بعد ما نرسلها؟",
    ],
    "⚠️ Pièges FR": [
        "Puis-je joindre un fichier de 10 Mo ?",
        "Puis-je réouvrir une réclamation après 10 jours ?",
        "Est-ce que le numéro de téléphone est obligatoire ?",
        "Puis-je soumettre une réclamation sans accepter les conditions ?",
        "Est-ce que la réclamation reste ouverte indéfiniment ?",
    ],
    "⚠️ Pièges AR": [
        "هل يمكنني رفع ملف حجمه 10 ميغا؟",
        "هل يمكن إعادة فتح الشكاية بعد 10 أيام؟",
        "هل رقم الهاتف إجباري؟",
        "هل يمكن تقديم شكاية بدون الموافقة على الشروط؟",
        "هل تبقى الشكاية مفتوحة إلى الأبد؟",
    ],
    "🧠 Scénario FR": [
        "J'ai soumis une réclamation mais je n'ai pas de réponse, que faire ?",
        "Je n'ai pas la référence de ma réclamation, comment la retrouver ?",
        "Je veux ajouter un document après envoi, est-ce possible ?",
        "J'ai dépassé 7 jours, puis-je encore répondre ?",
        "Je n'ai pas reçu le questionnaire de satisfaction, pourquoi ?",
    ],
    "🧠 Scénario AR": [
        "قدمت شكاية ولم أتلق أي رد، ماذا أفعل؟",
        "ضيعت رقم المرجع ديال الشكاية، كيفاش نلقاه؟",
        "بغيت نضيف وثيقة من بعد الإرسال، واش ممكن؟",
        "فاتت 7 أيام، واش باقي نقدر نجاوب؟",
        "ما توصلتش باستبيان الرضا، علاش؟",
    ],
    "🔍 Multi-étapes FR": [
        "Quelles sont les informations demandées dans la première étape ?",
        "Quelle est la différence entre les cas assurance et prévoyance ?",
        "Que contient la timeline d'une réclamation ?",
        "Quelles actions peut-on faire après consultation d'une réclamation ?",
    ],
    "🔍 Multi-étapes AR": [
        "ما هي المعلومات المطلوبة في المرحلة الأولى؟",
        "ما الفرق بين حالة التأمين وحالة الاحتياط؟",
        "ماذا تحتوي صفحة تتبع الشكاية؟",
        "ما هي العمليات التي يمكن القيام بها بعد عرض الشكاية؟",
    ],
    "🧩 Bruit FR+Darija mix": [
        "Ana bghit ndir reclamation fin نمشي ؟",
        "wach khasni email wla tel suffit ؟",
        "j'ai perdu la référence wach n9der nrecuperiha ?",
        "fin nلقى suivi dial dossier ?",
    ],
    "🧩 Bruit AR mix": [
        "بغيت نتابع الشكاية ديالي فين نمشي؟",
        "واش خاصني بجوج email ورقم الهاتف؟",
        "كيفاش نشوف الحالة ديال الشكاية؟",
        "واش نقدر نعاود نفتح الشكاية؟",
    ],
}


def ask(question: str) -> dict:
    body = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()}"}
    except Exception as e:
        return {"error": str(e)}


def grade(r: dict) -> str:
    if "error" in r:
        return "❌ ERROR"
    if r.get("metadata", {}).get("blocked"):
        return "🚫 BLOCKED"
    conf = r.get("confidence", 0)
    cits = len(r.get("citations", []))
    intent = r.get("metadata", {}).get("intent", "—")
    tag = "✅" if conf >= 0.40 else ("🔄" if conf == 0 else "⚠️")
    return f"{tag} conf={conf:.2f} cit={cits} intent={intent}"


def run():
    results = {}
    total = sum(len(v) for v in CATEGORIES.values())
    done = 0
    ok = blocked = errors = 0

    print(f"\n{'='*70}")
    print(f"  ATLAS-RAG PIPELINE TEST  —  {total} questions")
    print(f"{'='*70}\n")

    for cat, questions in CATEGORIES.items():
        print(f"\n{cat}")
        print("─" * 60)
        cat_results = []

        for q in questions:
            done += 1
            sys.stdout.write(f"  [{done:02d}/{total}] {q[:55]:<55} → ")
            sys.stdout.flush()
            t0 = time.time()
            r = ask(q)
            elapsed = time.time() - t0
            grade_str = grade(r)
            print(f"{grade_str}  ({elapsed:.1f}s)")

            if "error" in r:
                errors += 1
            elif r.get("metadata", {}).get("blocked"):
                blocked += 1
            else:
                ok += 1

            cat_results.append({
                "q": q,
                "grade": grade_str,
                "answer": r.get("answer", r.get("error", ""))[:200],
                "confidence": r.get("confidence"),
                "citations": len(r.get("citations", [])),
                "intent": r.get("metadata", {}).get("intent"),
                "elapsed": round(elapsed, 1),
            })

        results[cat] = cat_results

    print(f"\n{'='*70}")
    print(f"  RÉSULTATS  ✅ {ok}  🚫 {blocked}  ❌ {errors}  / {total}")
    print(f"{'='*70}")

    # Save full results
    out = "tests/test_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Résultats détaillés → {out}\n")


if __name__ == "__main__":
    run()

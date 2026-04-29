"""
Re-test only the 12 failed tests from the previous run.
Execution : docker exec atlas-api python /app/test_recheck.py
"""
import json, sys, time, urllib.request, urllib.error
 
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
 
API = "http://localhost:8080/chat"
TIMEOUT = 300
 
FAILED_TESTS = [
    ("F03", "fr", "IN_GUIDE",
     "Quelle est la taille maximale totale des pieces jointes ?",
     ["5"]),
    ("F04", "fr", "IN_GUIDE",
     "Quels formats de fichiers sont acceptes pour les pieces jointes ?",
     ["pdf", "word", "excel", "image"]),
    ("F06", "fr", "IN_GUIDE",
     "Peut-on rouvrir une reclamation apres l avoir cloturee ?",
     ["non", "impossible", "plus possible", "ne peut pas", "ne peut plus", "cloture", "clôture"]),
    ("F09", "fr", "IN_GUIDE",
     "Comment valider mon adresse email lors de la soumission ?",
     ["otp", "code", "6", "email"]),
    ("F12", "fr", "IN_GUIDE",
     "Comment acceder a la page de suivi detaillee d une reclamation ?",
     ["détails", "tableau", "rechercher"]),
    ("F14", "fr", "IN_GUIDE",
     "Quelles informations affiche le tableau de resultats de suivi ?",
     ["reference", "date", "statut", "motif", "type"]),
    ("F17", "fr", "IN_GUIDE",
     "Quelles sont les conditions pour demander la reouverture d une reclamation et combien de fois peut-on le faire ?",
     ["une fois", "motif", "message", "réouverture", "reouverture"]),
    ("F18", "fr", "IN_GUIDE",
     "Que contient la page de suivi detaillee d une reclamation ?",
     ["statut", "timeline", "echange", "réponse"]),
    ("F26", "fr", "OUT_OF_SCOPE",
     "Raconte-moi une blague.",
     None),
    ("A11", "ar", "IN_GUIDE",
     "ما هي المعلومات الواردة في صفحة المتابعة التفصيلية ؟",
     ["حالة", "مراسلات", "تاريخ"]),
    ("A15", "ar", "NOT_IN_GUIDE",
     "ما هو رقم هاتف ACAPS ؟",
     None),
    ("A18", "ar", "OUT_OF_SCOPE",
     "كيف اشتري سيارة في المغرب ؟",
     None),
]
 
NOT_IN_GUIDE_PHRASES = [
    "pas dans", "pas mentionn", "ne trouve pas", "pas d'information",
    "pas disponible", "n'est pas", "guide ne", "aucune information",
    "لا أجد", "غير موجود", "لا تتوفر", "ليست في", "لا يذكر",
    "لا يوجد", "لا توجد", "لم يرد", "غير مذكور", "لا يذكر ذلك",
    "لا تتضمن", "لم يتم ذكر", "غير متاح", "لا يتضمن",
    "ليس متاح", "ليس متوفر", "لا يتوفر",
    "not found", "not in", "cannot find",
    "n'est pas mentionn", "pas explicitement",
]
 
REDIRECT_PHRASES = [
    "uniquement", "seulement", "concu pour", "conçu pour",
    "portail acaps", "reformulez", "cadre",
    "فقط", "إعادة صياغة", "بوابة acaps",
]
 
 
def ask(question):
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(API, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"answer": f"[ERROR: {exc}]", "confidence": 0.0}
 
 
def evaluate(category, keywords, response_data):
    answer = response_data.get("answer", "")
    answer_lower = answer.lower()
    if "[ERROR" in answer:
        return "ERROR"
    if category == "IN_GUIDE":
        if not keywords:
            return "OK"
        hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        if hits >= max(1, len(keywords) // 2):
            return "OK"
        if any(p in answer_lower for p in NOT_IN_GUIDE_PHRASES):
            return "FAUX_NEGATIF"
        return f"INCOMPLET({hits}/{len(keywords)})"
    elif category == "NOT_IN_GUIDE":
        if any(p in answer_lower for p in NOT_IN_GUIDE_PHRASES):
            return "OK_REFUSE"
        return "INVENTE?"
    elif category == "OUT_OF_SCOPE":
        if any(p in answer_lower for p in REDIRECT_PHRASES + NOT_IN_GUIDE_PHRASES):
            return "OK_REDIRIGE"
        return "REPOND_HORS_SUJET"
    return "?"
 
 
COLORS = {
    "OK": "\033[92m", "OK_REFUSE": "\033[92m", "OK_REDIRIGE": "\033[92m",
    "FAUX_NEGATIF": "\033[91m", "INVENTE?": "\033[93m",
    "REPOND_HORS_SUJET": "\033[93m", "ERROR": "\033[91m", "RESET": "\033[0m",
}
 
 
def color(s):
    c = COLORS.get(s.split("(")[0], "")
    return f"{c}{s}{COLORS['RESET']}"
 
 
print(f"\n{'='*72}")
print(f"  RE-TEST — 12 cas echoues (chunk=700 chars + prompts renforces)")
print(f"{'='*72}\n")
 
print("  Pre-warmup query...")
t0 = time.time()
ask("Comment soumettre une reclamation ?")
print(f"  Warmup done ({time.time()-t0:.0f}s).\n")
 
results = []
for i, (tid, lang, cat, q, kw) in enumerate(FAILED_TESTS, 1):
    print(f"[{i:02d}/{len(FAILED_TESTS)}] {tid} [{lang.upper()}][{cat}]")
    print(f"     Q: {q[:80]}")
    t0 = time.time()
    resp = ask(q)
    elapsed = time.time() - t0
    status = evaluate(cat, kw, resp)
    preview = resp.get("answer", "")[:120].replace("\n", " ")
    conf = resp.get("confidence", 0.0)
    print(f"     A: {preview}...")
    print(f"     >> {color(status)}  conf={conf:.2f}  ({elapsed:.1f}s)\n")
    results.append((tid, status))
    sys.stdout.flush()
 
ok = sum(1 for _, s in results if s.startswith("OK"))
print(f"{'='*72}")
print(f"  RESULTAT : {ok}/{len(results)} corriges  (etaient tous ECHEC avant)")
print(f"{'='*72}")
for tid, status in results:
    mark = "✓" if status.startswith("OK") else "✗"
    print(f"  {mark} {tid}: {color(status)}")
print()
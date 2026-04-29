"""
Test chatbot contre les guides ACAPS.
Categorisation :
  IN_GUIDE   -> reponse attendue presente dans le guide (bot doit repondre correctement)
  NOT_IN_GUIDE -> information absente du guide (bot doit dire "pas dans le guide")
  OUT_OF_SCOPE -> hors perimetre ACAPS (bot doit rediriger)
 
Execution : docker exec atlas-api python /app/test_guide_limits.py
"""
import json
import sys
import time
import urllib.request
import urllib.error
 
# Force UTF-8 output on Windows to handle Arabic and special characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
 
API = "http://localhost:8080/chat"
TIMEOUT = 240
 
 
OLLAMA_RECOVERY_WAIT = 90  # seconds to wait after a connection reset before retrying
 
 
def ask(question: str) -> dict:
    payload = json.dumps({"question": question}).encode("utf-8")
    last_exc = None
    for _, wait_before in enumerate([0, OLLAMA_RECOVERY_WAIT]):
        if wait_before > 0:
            print(f"       [Server reset — waiting {wait_before}s for Ollama recovery...]", flush=True)
            time.sleep(wait_before)
        req = urllib.request.Request(
            API,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = exc
            break  # non-network errors: no point retrying
    return {"answer": f"[ERROR: {last_exc}]", "confidence": 0.0}
 
 
# ---------------------------------------------------------------------------
# Corpus de tests derive strictement des guides
# Format : (id, langue, categorie, question, mots_cles_attendus_ou_None)
#
# categorie :
#   IN_GUIDE      -> le guide contient la reponse
#   NOT_IN_GUIDE  -> information non mentionnee dans le guide
#   OUT_OF_SCOPE  -> sujet etranger au portail ACAPS
# ---------------------------------------------------------------------------
TESTS = [
    # ── Faits simples FR ────────────────────────────────────────────────────
    ("F01", "fr", "IN_GUIDE",
     "Quels types de reclamations peut-on soumettre via le portail ?",
     ["assurance", "prevoyance", "prévoyance"]),
 
    ("F02", "fr", "IN_GUIDE",
     "Combien de chiffres contient le code OTP envoye par email ?",
     ["6"]),
 
    ("F03", "fr", "IN_GUIDE",
     "Quelle est la taille maximale totale des pieces jointes ?",
     ["5"]),
 
    ("F04", "fr", "IN_GUIDE",
     "Quels formats de fichiers sont acceptes pour les pieces jointes ?",
     ["pdf", "word", "excel", "image"]),
 
    ("F05", "fr", "IN_GUIDE",
     "Combien de fois peut-on rouvrir une reclamation ?",
     ["une", "fois", "1"]),
 
    ("F06", "fr", "IN_GUIDE",
     "Peut-on rouvrir une reclamation apres l avoir cloturee ?",
     ["non", "impossible", "plus possible", "ne peut pas", "ne peut plus", "cloture", "clôture"]),
 
    ("F07", "fr", "IN_GUIDE",
     "Quand le questionnaire de satisfaction s affiche-t-il ?",
     ["cloture", "clôture", "consent", "accepte", "accepté"]),
 
    ("F08", "fr", "IN_GUIDE",
     "A quoi sert le numero de reference genere a la soumission ?",
     ["suivi", "reference", "référence"]),
 
    # ── Processus FR (complexite moyenne) ───────────────────────────────────
    ("F09", "fr", "IN_GUIDE",
     "Comment valider mon adresse email lors de la soumission ?",
     ["otp", "code", "6", "email"]),
 
    ("F10", "fr", "IN_GUIDE",
     "Quelles informations personnelles faut-il renseigner a l etape 1 ?",
     ["nom", "email", "telephone", "identite", "adresse"]),
 
    ("F11", "fr", "IN_GUIDE",
     "Quelles informations sont necessaires pour suivre une reclamation ?",
     ["référence", "email", "téléphone"]),
 
    ("F12", "fr", "IN_GUIDE",
     "Comment acceder a la page de suivi detaillee d une reclamation ?",
     ["détails", "tableau", "rechercher"]),
 
    ("F13", "fr", "IN_GUIDE",
     "Comment repondre a l ACAPS depuis la page de suivi ?",
     ["message", "formulaire", "repondre", "répondre"]),
 
    ("F14", "fr", "IN_GUIDE",
     "Quelles informations affiche le tableau de resultats de suivi ?",
     ["reference", "date", "statut", "motif", "type"]),
 
    ("F15", "fr", "IN_GUIDE",
     "Quelles sont les deux options offertes apres la reponse definitive de l ACAPS ?",
     ["cloture", "clôturer", "reouverture", "réouverture"]),
 
    # ── Questions complexes FR ───────────────────────────────────────────────
    ("F16", "fr", "IN_GUIDE",
     "Decrivez toutes les etapes pour soumettre une reclamation assurance de l etape 0 a la confirmation",
     ["etape", "étape", "0", "1", "2", "3", "4", "5"]),
 
    ("F17", "fr", "IN_GUIDE",
     "Quelles sont les conditions pour demander la reouverture d une reclamation et combien de fois peut-on le faire ?",
     ["une fois", "motif", "message", "réouverture", "reouverture"]),
 
    ("F18", "fr", "IN_GUIDE",
     "Que contient la page de suivi detaillee d une reclamation ?",
     ["statut", "timeline", "echange", "réponse"]),
 
    # ── Limites : hors guide FR ──────────────────────────────────────────────
    ("F19", "fr", "NOT_IN_GUIDE",
     "Quel est le delai de traitement d une reclamation ?",
     None),
 
    ("F20", "fr", "NOT_IN_GUIDE",
     "Y a-t-il des frais pour soumettre une reclamation ?",
     None),
 
    ("F21", "fr", "NOT_IN_GUIDE",
     "Quel est le numero de telephone de l ACAPS ?",
     None),
 
    ("F22", "fr", "NOT_IN_GUIDE",
     "Peut-on soumettre une reclamation par courrier postal ?",
     None),
 
    ("F23", "fr", "NOT_IN_GUIDE",
     "L application mobile est-elle disponible sur iOS et Android ?",
     None),
 
    # ── Hors perimetre FR ────────────────────────────────────────────────────
    ("F24", "fr", "OUT_OF_SCOPE",
     "Quelle est la meteo a Casablanca ?",
     None),
 
    ("F25", "fr", "OUT_OF_SCOPE",
     "Comment acheter une assurance automobile au Maroc ?",
     None),
 
    ("F26", "fr", "OUT_OF_SCOPE",
     "Raconte-moi une blague.",
     None),
 
    # ── Faits simples AR ────────────────────────────────────────────────────
    ("A01", "ar", "IN_GUIDE",
     "ما هي انواع المطالبات التي يمكن تقديمها عبر البوابة ؟",
     ["تأمين", "توقع"]),
 
    ("A02", "ar", "IN_GUIDE",
     "كم عدد ارقام رمز التحقق OTP ؟",
     ["6"]),
 
    ("A03", "ar", "IN_GUIDE",
     "ما هو الحجم الاقصى للمرفقات ؟",
     ["5"]),
 
    ("A04", "ar", "IN_GUIDE",
     "ما هي صيغ الملفات المقبولة للارفاق ؟",
     ["pdf", "word", "excel"]),
 
    ("A05", "ar", "IN_GUIDE",
     "كم مرة يمكن اعادة فتح المطالبة ؟",
     ["مرة", "واحدة", "1"]),
 
    ("A06", "ar", "IN_GUIDE",
     "هل يمكن اعادة فتح مطالبة بعد اغلاقها ؟",
     ["لا", "غير ممكن", "مستحيل"]),
 
    ("A07", "ar", "IN_GUIDE",
     "متى يظهر استبيان الرضا ؟",
     ["اغلاق", "إغلاق", "وافق", "قبل"]),
 
    ("A08", "ar", "IN_GUIDE",
     "ما هو الرقم المرجعي وما فائدته ؟",
     ["متابعة", "فريد", "مرجع"]),
 
    # ── Processus AR ────────────────────────────────────────────────────────
    ("A09", "ar", "IN_GUIDE",
     "كيف يمكنني متابعة حالة مطالبتي ؟",
     ["رقم", "مرجع", "otp", "بحث"]),
 
    ("A10", "ar", "IN_GUIDE",
     "ما هي المعلومات المطلوبة لتتبع مطالبة ؟",
     ["مرجع", "بريد", "هاتف"]),
 
    ("A11", "ar", "IN_GUIDE",
     "ما هي المعلومات الواردة في صفحة المتابعة التفصيلية ؟",
     ["حالة", "مراسلات", "تاريخ"]),
 
    ("A12", "ar", "IN_GUIDE",
     "كيف يمكنني الرد على ACAPS من صفحة المتابعة ؟",
     ["رسالة", "نموذج", "رد"]),
 
    # ── Questions complexes AR ───────────────────────────────────────────────
    ("A13", "ar", "IN_GUIDE",
     "ما هي جميع الخطوات اللازمة لتقديم مطالبة تامين ؟",
     ["خطوة", "0", "1", "2"]),
 
    ("A14", "ar", "IN_GUIDE",
     "ما هي شروط طلب اعادة فتح المطالبة ؟",
     ["سبب", "رسالة", "مرة واحدة"]),
 
    # ── Limites hors guide AR ────────────────────────────────────────────────
    ("A15", "ar", "NOT_IN_GUIDE",
     "ما هو رقم هاتف ACAPS ؟",
     None),
 
    ("A16", "ar", "NOT_IN_GUIDE",
     "هل هناك رسوم لتقديم المطالبة ؟",
     None),
 
    # ── Hors perimetre AR ────────────────────────────────────────────────────
    ("A17", "ar", "OUT_OF_SCOPE",
     "ما هي درجة الحرارة في الرباط اليوم ؟",
     None),
 
    ("A18", "ar", "OUT_OF_SCOPE",
     "كيف اشتري سيارة في المغرب ؟",
     None),
]
 
# ---------------------------------------------------------------------------
# Evaluateur simple
# ---------------------------------------------------------------------------
NOT_IN_GUIDE_PHRASES = [
    "pas dans", "pas mentionn", "ne trouve pas", "pas d'information",
    "pas disponible", "n'est pas", "guide ne", "aucune information",
    "لا أجد", "غير موجود", "لا تتوفر", "ليست في", "لا يذكر",
    "لا يوجد", "لا توجد", "لم يرد", "غير مذكور", "لا يذكر ذلك",
    "لا تتضمن", "لم يتم ذكر", "غير متاح", "لا يتضمن",
    "ليس متاح", "ليس متوفر", "لا يتوفر",
    "not found", "not in", "cannot find",
]
 
REDIRECT_PHRASES = [
    "uniquement", "seulement", "concu pour", "conçu pour",
    "portail acaps", "reformulez", "cadre",
    "فقط", "إعادة صياغة", "بوابة acaps",
]
 
 
def evaluate(test_id, lang, category, question, keywords, response_data) -> str:
    answer = response_data.get("answer", "")
    confidence = response_data.get("confidence", 0.0)
    answer_lower = answer.lower()
 
    if "[ERROR" in answer or "[CONNECTION" in answer:
        return "ERROR"
 
    if category == "IN_GUIDE":
        if not keywords:
            return "OK"
        hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        if hits >= max(1, len(keywords) // 2):
            return "OK"
        # Check if bot said "not in guide" for something that IS in guide
        if any(p in answer_lower for p in NOT_IN_GUIDE_PHRASES):
            return "FAUX_NEGATIF"
        return f"INCOMPLET({hits}/{len(keywords)})"
 
    elif category == "NOT_IN_GUIDE":
        if any(p in answer_lower for p in NOT_IN_GUIDE_PHRASES):
            return "OK_REFUSE"
        # Bot answered with content — but it shouldn't have info in guide
        return "INVENTE?"
 
    elif category == "OUT_OF_SCOPE":
        if any(p in answer_lower for p in REDIRECT_PHRASES + NOT_IN_GUIDE_PHRASES):
            return "OK_REDIRIGE"
        return "REPOND_HORS_SUJET"
 
    return "?"
 
 
# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
COLORS = {
    "OK": "\033[92m",         # green
    "OK_REFUSE": "\033[92m",
    "OK_REDIRIGE": "\033[92m",
    "FAUX_NEGATIF": "\033[91m",  # red
    "INVENTE?": "\033[93m",   # yellow
    "REPOND_HORS_SUJET": "\033[93m",
    "ERROR": "\033[91m",
    "RESET": "\033[0m",
}
 
 
def color(status: str) -> str:
    c = COLORS.get(status.split("(")[0], "")
    return f"{c}{status}{COLORS['RESET']}"
 
 
def warmup():
    """Send a real query to load the Ollama model; retry up to 3 times on failure."""
    print("  Warming up LLM (loading model into memory)...")
    for attempt in range(3):
        t0 = time.time()
        resp = ask("Comment soumettre une reclamation sur le portail ?")
        elapsed = time.time() - t0
        answer = resp.get("answer", "")
        if "[ERROR" not in answer and "[CONNECTION" not in answer:
            print(f"  Warmup done (attempt {attempt + 1}, {elapsed:.0f}s).\n")
            return
        print(f"  Warmup attempt {attempt + 1} failed ({elapsed:.0f}s) — retrying...")
        time.sleep(30)
    print("  Warmup failed after all attempts — proceeding anyway.\n")
 
 
def run():
    results = []
    total = len(TESTS)
    print(f"\n{'='*72}")
    print(f"  TEST CHATBOT vs GUIDES ACAPS  ({total} questions)")
    print(f"{'='*72}\n")
    warmup()
 
    for i, (tid, lang, category, question, keywords) in enumerate(TESTS, 1):
        print(f"[{i:02d}/{total}] {tid} [{lang.upper()}][{category}]")
        print(f"     Q: {question[:80]}")
        t0 = time.time()
        resp = ask(question)
        elapsed = time.time() - t0
        status = evaluate(tid, lang, category, question, keywords, resp)
        answer_preview = resp.get("answer", "")[:120].replace("\n", " ")
        conf = resp.get("confidence", 0.0)
        print(f"     A: {answer_preview}...")
        print(f"     >> {color(status)}  conf={conf:.2f}  ({elapsed:.1f}s)\n")
        results.append((tid, lang, category, status))
        sys.stdout.flush()
 
    # Summary
    print(f"\n{'='*72}")
    print("  SYNTHESE")
    print(f"{'='*72}")
    cats = {}
    for tid, lang, cat, status in results:
        cats.setdefault(cat, []).append(status)
 
    total_ok = 0
    total_all = 0
    for cat, statuses in sorted(cats.items()):
        ok = sum(1 for s in statuses if s.startswith("OK"))
        total_ok += ok
        total_all += len(statuses)
        print(f"  {cat:<16} {ok}/{len(statuses)} corrects")
        for s in sorted(set(statuses)):
            count = statuses.count(s)
            print(f"    {color(s):<40} x{count}")
 
    print(f"\n  TOTAL: {total_ok}/{total_all} ({100*total_ok//total_all}%)\n")
 
    # Failures detail
    failures = [(tid, lang, cat, st) for tid, lang, cat, st in results
                if not st.startswith("OK") and st != "ERROR"]
    if failures:
        print("  ECHECS A CORRIGER:")
        for tid, lang, cat, st in failures:
            q = next(t[3] for t in TESTS if t[0] == tid)
            print(f"    {tid} [{lang}] {st}: {q[:70]}")
    print()
 
 
if __name__ == "__main__":
    run()
 
 
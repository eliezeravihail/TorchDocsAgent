# TorchDocs Agent — תוכנית ביצוע מפורטת (רמת TODO)

מסמך עבודה לביצוע. כל משימה מנוסחת כך שג'וניור יכול לבצע אותה, עם **קריטריון קבלה** ("סיימת כאשר...") והערכת זמן.
משימות מסומנות `[CORE]` הן חובה; `[STRETCH]` — רק אם נשאר זמן. אל תתחיל STRETCH לפני שכל ה-CORE של אותה אבן דרך ירוק.

**החלטות מחייבות (לא לפתוח מחדש בזמן הביצוע):**
- גרסת PyTorch נעוצה: **torch 2.7.x** — האינדקס, ה-sandbox וה-eval כולם על אותה גרסה.
- היקף הקורפוס: **רק** `torch/nn`, `torch/optim`, `torch/utils/data`, `torch/nn/functional`, `torch/autograd` + הדוקס הרשמיים שלהם. לא C++, לא CUDA, לא internals.
- **אחסון pointer-based:** ה-DB לא מאחסן קוד גולמי — רק embeddings, tsvector ו-metadata (נתיב, שורות, סימבול, חתימה). מקור האמת היחיד הוא ה-clone הנעוץ; התוכן נקרא ממנו בזמן שאילתה (hydrate).
- שפה: Python 3.11+. כל קריאות ה-LLM דרך LiteLLM מהיום הראשון של M3 (עד אז — SDK ישיר).

---

## M0 · הקמה (יום–יומיים)

- [ ] [CORE] ריפו חדש `torchdocs-agent` עם המבנה מה-README (`ingest/`, `index/`, `agent/`, `eval/`, `app/`), `pyproject.toml`, `ruff`, `pytest`, pre-commit.
  ✔ סיימת כאשר: `pytest` רץ ירוק על בדיקת placeholder אחת.
- [ ] [CORE] חשבונות: Neon (פרויקט + DB), מפתח LLM אחד לפחות (Anthropic/OpenAI), Langfuse cloud (או דחיית self-host ל-M4).
  ✔ סיימת כאשר: `psql $NEON_URL -c "select 1"` עובד ו-`.env.example` קיים בריפו.
- [ ] [CORE] סקריפט `scripts/smoke.py`: קריאת LLM אחת + כתיבה/קריאה מ-Neon.
  ✔ סיימת כאשר: הסקריפט רץ נקי משורת הפקודה.

---

## M1 · הגרעין המחולל (שבועות 1–2)

### 1.1 סכמת פלט
- [ ] [CORE] `agent/schemas.py`: מודל Pydantic `CodeAnswer` עם שדות `code: str`, `explanation: str`, `symbols_used: list[str]`, `torch_version: str`.
  ✔ סיימת כאשר: יש בדיקת round-trip (dict → model → dict) עוברת.

### 1.2 עטיפת LLM
- [ ] [CORE] `agent/llm.py`: פונקציה `generate_code(question: str) -> CodeAnswer` עם structured output, retry (עד 3, exponential backoff), ו-timeout.
  ✔ סיימת כאשר: 10 שאלות שונות מחזירות `CodeAnswer` תקין בלי חריגות.
- [ ] [CORE] טיפול בכשל parsing: אם הפלט לא נכנס לסכמה — ניסיון תיקון אחד עם הודעת השגיאה, אחרת החזרת שגיאה מסודרת.
  ✔ סיימת כאשר: בדיקה עם mock שמחזיר JSON שבור עוברת.

### 1.3 eval ראשון — מהיום הראשון
- [ ] [CORE] `eval/checks.py`: שלוש בדיקות על כל `CodeAnswer`: (א) `ast.parse` מצליח; (ב) כל `import` הוא torch/סטנדרטי; (ג) כל סימבול ב-`symbols_used` באמת מופיע בקוד.
  ✔ סיימת כאשר: הבדיקות רצות על 10 תשובות ומדפיסות טבלת pass/fail.
- [ ] [CORE] `eval/questions_v0.jsonl`: 15 שאלות PyTorch ידניות (5 קלות: "מה עושה nn.Dropout"; 5 בינוניות: "כתוב DataLoader עם sampler מותאם"; 5 קשות: "custom autograd Function").
  ✔ סיימת כאשר: הקובץ קיים וסקריפט `eval/run_v0.py` מריץ את כולן ושומר תוצאות.
- [ ] [CORE] **תיעוד ההזיות**: הרץ את 15 השאלות, עבור ידנית על הקוד, ורשום ב-`eval/hallucinations.md` כל API שהומצא או חתימה שגויה.
  ✔ סיימת כאשר: יש לפחות 3 דוגמאות מתועדות. *(זו ההצדקה המדידה ל-M2 — אל תדלג.)*

**Gate ל-M2:** יש מחולל עובד + סט 15 שאלות + רשימת הזיות מתועדת.

---

## M2 · ההארקה (שבועות 3–4)

### 2.1 Ingestion
- [ ] [CORE] `ingest/clone.py`: הורדת torch 2.7 (tag נעוץ, `--depth 1`) וסינון לתיקיות שבהיקף בלבד.
  ✔ סיימת כאשר: תיקיית `_corpus/` מכילה רק קבצי המודולים שבהיקף (סדר גודל מאות קבצים, לא אלפים).
- [ ] [CORE] `ingest/chunk_code.py`: חיתוך קוד לפי מבנה עם מודול `ast` — chunk לכל פונקציה/מחלקה, עם metadata: `file_path`, `start_line`, `end_line`, `symbol_name`, docstring.
  ✔ סיימת כאשר: על `torch/nn/modules/linear.py` מתקבלים chunks נפרדים ל-`Linear`, `Bilinear` וכו', עם שורות נכונות.
- [ ] [CORE] `ingest/chunk_docs.py`: חיתוך קבצי rst/markdown של הדוקס לפי כותרות, אותה סכמת metadata.
  ✔ סיימת כאשר: מדגם 5 קבצים נחתך הגיוני בבדיקה ידנית.

### 2.2 אינדוקס ב-Neon
- [ ] [CORE] סכמת טבלה: `chunks(id, embedding vector, tsv tsvector, file_path, start_line, end_line, symbol_name, signature, kind)` — **בלי עמודת תוכן גולמי**. ה-tsvector מחושב בזמן האינדוקס (מהתוכן, שנקרא ואינו נשמר) ומספיק לחיפוש מילות-מפתח. אינדקס HNSW על embedding + GIN על tsv.
  ✔ סיימת כאשר: מיגרציה רצה נקי, ו-`select * from chunks limit 1` לא מכיל שום קוד — רק וקטורים ו-metadata.
- [ ] [CORE] `index/embed.py`: חישוב embeddings ב-batches (עם עמידות לכשל אמצע-דרך — שמירת checkpoint), והכנסה ל-Neon.
  ✔ סיימת כאשר: כל הקורפוס מאונדקס; שאילתת `count(*)` הגיונית; הרצה חוזרת לא מכפילה שורות.

### 2.3 אחזור היברידי
- [ ] [CORE] `index/retrieve.py`: פונקציה `retrieve(query, k=8)` שממזגת dense (pgvector) + מילות-מפתח (tsvector), עם דירוג RRF פשוט. מחזירה **pointers** (נתיב + טווח שורות), לא תוכן.
  ✔ סיימת כאשר: החיפוש `scaled_dot_product_attention` מחזיר את ה-pointer להגדרה האמיתית בתוצאה הראשונה (dense לבד נכשל בזה — זו הבדיקה).
- [ ] [CORE] `index/hydrate.py`: קריאת השורות בפועל מה-clone הנעוץ לפי ה-pointers, לקראת הזרקה לפרומפט.
  ✔ סיימת כאשר: hydrate על תוצאת retrieve מחזיר בדיוק את הפונקציה, ובדיקה מוודאת התאמה בין ה-metadata לתוכן בקובץ.
- [ ] [STRETCH] reranker (cross-encoder קטן או LLM-rerank) מעל ה-top-20.

### 2.4 חיבור והערכה
- [ ] [CORE] עדכון `generate_code`: retrieve → hydrate → הזרקת הקטעים לפרומפט עם הוראה מפורשת "השתמש רק ב-APIs שמופיעים בקונטקסט", והוספת `citations: list[{file_path, lines}]` לסכמה.
  ✔ סיימת כאשר: התשובות כוללות ציטוטים אמיתיים שניתן לפתוח בקובץ.
  *הערה: מכאן ה-clone הנעוץ הוא dependency של זמן-ריצה — הוא ייכנס ל-image של ה-deploy ב-M5.*
- [ ] [CORE] מדד ייעודי `grounded_api_rate`: אחוז הסימבולים ב-`symbols_used` שקיימים באינדקס. הרצה על 15 השאלות מ-M1, השוואה לפני/אחרי RAG.
  ✔ סיימת כאשר: יש טבלה אחת שמראה את השיפור — זה גם חומר מעולה ל-README.
- [ ] [STRETCH] RAGAS על סט השאלות (context precision/recall, faithfulness).

**Gate ל-M3:** `grounded_api_rate` השתפר משמעותית מול M1, וההזיות מ-`hallucinations.md` נעלמו או פחתו.

---

## M3 · הסוכן (שבועות 5–6)

### 3.1 sandbox להרצת קוד
- [ ] [CORE] `agent/runner.py`: הרצת קוד ב-subprocess מבודד — Docker image עם torch 2.7 **CPU-only** (חוסך GB ומכונת GPU), timeout 30 שניות, הגבלת זיכרון, ללא רשת.
  ✔ סיימת כאשר: קוד תקין מחזיר stdout; לולאה אינסופית נהרגת ב-timeout; `import requests` נכשל.
  *הערה: להתחיל לוקאלית עם Docker. מעבר ל-Modal — רק ב-M5.*

### 3.2 הלולאה הידנית
- [ ] [CORE] `agent/loop.py`: agent loop ידני (יעד ~150 שורות): תכנון → retrieve → generate → run → אם שגיאה: הזרקת ה-traceback חזרה ותיקון (עד 3 סבבים) → תשובה עם ציטוטים.
  ✔ סיימת כאשר: "בנה training loop עם mixed precision" עובר את כל המסלול ומחזיר קוד שרץ.
- [ ] [CORE] self-grade על ה-retrieval: אחרי retrieve, קריאת LLM קצרה ששופטת אם הקונטקסט מספיק; אם לא — ניסוח מחדש של השאילתה וניסיון נוסף (פעם אחת).
  ✔ סיימת כאשר: יש בדיקה עם שאלה מעורפלת שמדגימה שכתוב שאילתה.

### 3.3 LiteLLM gateway
- [ ] [CORE] העברת כל הקריאות דרך LiteLLM proxy עם config: ספק ראשי + fallback, תקציב יומי, ותיוג כל קריאה (`m3-loop`, `m3-grade`...).
  ✔ סיימת כאשר: דוח עלות per-request מופיע בלוגים של LiteLLM.

### 3.4 LangGraph והשוואה
- [ ] [CORE] שכתוב הלולאה כ-LangGraph graph (אותם צמתים בדיוק).
  ✔ סיימת כאשר: שתי הגרסאות עוברות את אותו סט 15 שאלות עם תוצאות דומות.
- [ ] [CORE] `docs/loop-vs-langgraph.md`: השוואה קצרה — שורות קוד, קלות debugging, latency. עמוד אחד.
- [ ] [STRETCH] חשיפת retrieve + runner כשרתי MCP עם FastMCP; בדיקה מלקוח MCP.
- [ ] [STRETCH] routing בין מסלול "הסבר" (בלי runner) ל"בנייה" (עם runner).
- [ ] [STRETCH] זיכרון long-term (העדפות משתמש, גרסת torch) — לדחות אם אין זמן.

**Gate ל-M4:** בקשת בנייה אמיתית עוברת plan→retrieve→generate→run→fix→cite מקצה לקצה.

---

## M4 · המשמעת (שבוע 7)

- [ ] [CORE] חיבור Langfuse: trace לכל הרצה עם span לכל צעד (plan / retrieve / generate / run / fix).
  ✔ סיימת כאשר: אפשר לפתוח הרצה כושלת ב-UI ולראות באיזה צעד היא נשברה.
- [ ] [CORE] הרחבת סט ההערכה ל-**40 שאלות** ב-`eval/questions_v1.jsonl`, לכל אחת: שאלה, סוג (explain/build), ותשובת-זהב או assertion אוטומטי (למשל: "הקוד חייב להריץ forward על tensor 2x3 בלי חריגה").
  ✔ סיימת כאשר: `eval/run_v1.py` מריץ את כולן ומדפיס: pass rate, grounded_api_rate, executability rate, עלות ו-latency ממוצעים.
- [ ] [CORE] טקסונומיית שגיאות: סיווג כל כישלון לאחת מ-4 קטגוריות (API מזויף / החמצת retrieval / שגיאת ריצה / ציטוט שגוי), רישום ב-MLflow, ו-`docs/error-analysis.md` עם 3 מסקנות ופעולה אחת לשיפור שבוצעה בפועל.
  ✔ סיימת כאשר: יש לפני/אחרי מדיד של שיפור אחד לפחות.
- [ ] [CORE] cache ב-Upstash Redis: exact-match על (שאלה, גרסת אינדקס) לתשובות, ו-cache ל-embeddings של שאילתות.
  ✔ סיימת כאשר: שאלה חוזרת חוזרת מה-cache ב-<200ms, ונמדד hit-rate.
- [ ] [STRETCH] semantic cache (דמיון וקטורי בין שאלות) — רק אחרי שה-exact cache עובד.

**Gate ל-M5:** דוח eval אחד מלא + trace אחד שאפשר להראות בראיון.

---

## M5 · שילוח + hardening (שבוע 8)

- [ ] [CORE] ממשק Gradio מינימלי: שדה שאלה, תשובה עם קוד מודגש, ציטוטים לחיצים, ואינדיקציית "הקוד רץ ✓".
- [ ] [CORE] פריסה על שכבה חינמית — בחירה אחת: HF Spaces (הכי מהיר) או Modal (מרשים יותר, כולל ה-sandbox) או Railway.
  ✔ סיימת כאשר: לינק ציבורי עובד מדפדפן נקי, כולל שאילתא מלאה.
- [ ] [CORE] auth בסיסי: API key לכל משתמש (טבלה ב-Neon), rate limit per-key, וכל הרצת קוד מתויגת ל-key.
  ✔ סיימת כאשר: בקשה בלי key נדחית; key אחד לא יכול לחרוג מהמכסה.
  *הערה לג'וניור: לא OAuth מלא. API keys מספיקים להוכחת multi-user.*
- [ ] [CORE] תקרות עלות: תקציב per-key ותקרה יומית גלובלית דרך LiteLLM; חריגה מחזירה שגיאה מסודרת.
- [ ] [CORE] secrets ב-Infisical (או secrets manager של פלטפורמת ה-deploy) — אפס סודות בקוד.
- [ ] [CORE] עדכון README: צילומי מסך, טבלת תוצאות ה-eval, לינק חי.
  ✔ סיימת כאשר: אדם זר יכול להבין את הפרויקט ולנסות אותו תוך 2 דקות.
- [ ] [STRETCH] דף "cost story": כמה עולה שאילתא ממוצעת, ואיפה ה-free tier נגמר.

---

## הרחבות עתידיות (לא בטווח 8 השבועות)
- קליטת צילום traceback/דיאגרמה (VLM/OCR).
- קורפוס שני: libtorch C++ או docs-site JS.
- ערוץ WhatsApp/Slack כ-frontend נוסף (אותו agent, מעטפת ערוץ).

## כללי עצירה (חשוב לג'וניור)
- נתקעת מעל חצי יום על משימת CORE? צמצם את ההיקף שלה ותעד את הקיצוץ — אל תרחיב את הזמן.
- כל אבן דרך נסגרת עם commit מתויג (`m1-done`...) ושורת סיכום ב-README.
- אין לגעת ב-STRETCH כשה-CORE אדום. אין להוסיף features שלא ברשימה.

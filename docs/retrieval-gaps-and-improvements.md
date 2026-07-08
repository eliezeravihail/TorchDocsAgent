---
kind: gap-analysis
date: 2026-07-08
scope: retrieval quality + eval depth (RAG maturity review)
evidence: [eval/results/retrieval_v1.jsonl, eval/diagnose_retrieval.py,
  eval/results/agentic_v1_first6.jsonl, PLAN.md]
verdict: >
  Retrieval is the system's ceiling. The top gap is asymmetric dense
  retrieval on descriptive questions — MEASURED as the expected pages sitting
  at true dense rank 899-2999 inside their own kind-pool, far beyond any
  rerankable candidate set. The fix in flight is index-side contextual
  glosses (merged, generation running); a reranker is the precision stage
  AFTER candidate generation is fixed, not the fix itself. Second gap: no
  automated answer-quality eval (only retrieval metrics + static checks).
open_gaps: [answer-quality-eval, reranker-as-precision-stage,
  query-transformation, context-precision, refusal-calibration]
---

# חוסרים בשליפה ואיכות — ניתוח פערים והצעות שיפור

מסמך זה מסכם את הפערים המרכזיים במערכת ה־RAG הנוכחית של TorchDocsAgent
ומציע שיפורים מדורגים לפי מנוף. הבסיס הקיים מוצק — hybrid retrieval עם
per-kind pools ו־RRF, guard מבוסס מרחב־אמבדינג, grounding contract, static
checks, tool loop + LangGraph, ו־eval עם recall/MRR. הפערים כאן הם השכבה
הבאה, לא תיקון של יסודות שבורים.

## 1. הפער המרכזי — שליפה a-סימטרית על שאלות תיאוריות

### הראיה (מדוד, לא משוער)

מתוך `eval/results/retrieval_v1.jsonl`:

| id  | שאלה                                                              | recall |
|-----|-------------------------------------------------------------------|--------|
| c20 | "What does BCEWithLogitsLoss compute, and why more stable...?"     | 0.0    |
| c17 | "How does NLL loss compute the loss from log-probabilities?"       | 0.0    |
| c16 | "What does max pooling compute, including how gradients route?"    | 0.0    |
| c18 | "What does the softmax function compute and over which dimension?" | 1.0    |
| c19 | "How does torch.gather select values along a dimension?"           | 1.0    |

ההערה שכבר תועדה ב־`eval/diagnose_retrieval.py`:
> *"all descriptive, none containing the symbol token."*

### הדיאגנוזה

זו **בעיית ה־asymmetry** של dense retrieval — לא באג בכיול:

- **השאלה** תיאורית ועשירה: *"מה זה מחשב, למה יציב יותר"*.
- **עמוד ה־API** טרמינולוגי וקצר: `BCEWithLogitsLoss` + חתימה.
- dense מודד דמיון query↔doc, אבל כאן ה־query וה־doc **לא דומים — הם
  משלימים**. הם חיים באזורים שונים של מרחב האמבדינג.
- כשהשאלה כן מכילה את הטוקן (`softmax`, `gather`) — recall קופץ ל־1.0.
  זה מאשש שהבעיה היא vocabulary/style mismatch, לא כיסוי הקורפוס.

**עומק הפער — נמדד (Diagnose retrieval, 2026-07-08).** הרנק האמיתי של הדף
הצפוי בין כל צ׳אנקי ה־api, לפי מרחק קוסינוס מדויק:

| שאלה | דף צפוי | מרחק הצ׳אנק הקרוב | רנק dense אמיתי ב־api |
|---|---|---|---|
| a06 | torch.nn.Linear | 0.498 | **2,999** |
| a17 | CrossEntropyLoss | 0.464 | **1,743** |
| a10 | LayerNorm | 0.415 | **899** |
| c02 | torch.optim.SGD | 0.355 | **7** |

המספרים האלה מכריעים את בחירת המנוף (ראו בטבלה למטה): דף ברנק 1,743 לא
נמצא בשום קבוצת מועמדים ש־reranker סביר יקבל — **reranker רק מסדר מחדש
מועמדים שנשלפו**; הוא לא יכול להציל דף שלא נשלף. לעומת זאת SGD ברנק 7
חשף באג נפרד לגמרי (ראו "ממצא אינדקס" למטה).

**עיקרון על:** ב־RAG ה־retrieval הוא התקרה. אם העמוד הנכון לא ב־top-k,
שום חוכמה של המחולל לא תשחזר אותו. הלולאה האג׳נטית ממסכת חלק מזה עם ניסוח
מחדש — נמדד: agentic coverage ‎0.567‎ מול single-shot ‎0.133‎ (דלתא ‎+0.433‎,
`eval/results/agentic_v1_first6.jsonl`) — אבל לא יכולה לרנדר עמוד שאף
שאילתה לא מביאה.

### ממצא אינדקס נלווה (תוקן): HNSW post-filtering

‏SGD ברנק אמיתי 7 בתוך api — ובכל זאת נעדר מהטופ-20 של השאילתה המסוננת.
הסיבה: pgvector מפעיל את סינון ה־`WHERE` **אחרי** סריקת ה־HNSW המקורבת;
ב־`ef_search=40` (ברירת מחדל) ~40 השכנים הגלובליים של שאלה תיאורית הם רובם
טוטוריאלים, ודף ה־api נזרק לפני שהסינון רואה אותו. תוקן ב־#76
(`SET hnsw.ef_search = 150` בכל שאילתת pool). זה מציל דפים שרנקם בתוך
ה־pool — לא את אלה שברנק מאות-אלפים.

### המנופים לפתרון (מדורג מחדש לפי המדידה)

| מנוף | מה עושה | מצב היום |
|---|---|---|
| **גלוסות קונטקסטואליות (Contextual Retrieval)** | משפט הקשר בשפה טבעית מוקדם לכל צ׳אנק לפני ההטבעה (וגם ל־tsvector) — מזיז את הדף עצמו לאזור של השאלות; הכלי היחיד שמתקן רנק 899–2,999 | **מוזג (#71); ייצור הגלוסות רץ; מדידה אחרי re-embed** |
| **רוחב סריקת HNSW** | ראו "ממצא אינדקס" | **תוקן (#76)** |
| **Reranker (cross-encoder)** | קורא query+doc יחד ומדרג מחדש — שלב ה־**precision** אחרי שהדף בכלל נכנס למועמדים; לא מתקן ABSENT-from-pool | שלב הבא אחרי מדידת הגלוסות |
| **HyDE / query transformation** | LLM כותב תשובה משוערת ומאמבד אותה — צד-שאילתה משלים לגלוסות (צד-מסמך) | לא קיים; לשקול אם הגלוסות לא סוגרות את הפער |
| **Embedding גדול/מכוונן** | bge-base נוסה על v0 (סט קטן מדי — לא ראיה); התשתית להחלפה מדודה על v1 קיימת | פרמטרול מוזג (#70); קלף בקנה |

**המלצה מיידית (מעודכן):** למדוד את הגלוסות (re-embed → `eval-retrieval`
מול baseline‏ 0.430/0.345), ואז להחליט: אם רנק הדפים ירד לתוך ה־pool —
reranker הוא הצעד הבא הטבעי כשלב precision; אם לא — HyDE לפני reranker.

## 2. אין eval אוטומטי לאיכות התשובה (רק ל־retrieval)

**קיים:** recall/MRR ל־retrieval, ו־static checks (`eval/checks.py`: parse,
imports, symbols-in-index).

**נסגר (2026-07-08):** נוסף LLM-as-judge על **הנתיב הגראונדד** —
`eval/run_judge.py`, כפתור `suite=judge` ב־`Eval`. שלושה ציונים ‎[0,1]
לכל שאלה: **faithfulness** (כל טענה נתמכת ב־context שהוצג),
**answer-relevance**, **citation-correctness**. השופט רואה את *אותו* context
ממוספר שהמחולל ראה (לא re-retrieval), התוצאות נשמרות ל־`eval/results/judge_*`
עם aggregate before/after. RAGAS נשאר STRETCH.

**למה זה חשוב:** בלי faithfulness / answer-correctness אוטומטי, רגרסיות
מתגלות רק ידנית. זה הפער בין "פרויקט מרשים" ל"מערכת שאפשר לתחזק". **ה־eval
הוא המוצר.**

**מה שנותר:** (א) להריץ baseline ולתעד מספרים; (ב) מגבלה מדועת — כשהשופט
הוא אותו מודל חינמי שכתב את התשובה יש הטיית סלחנות, אז להצביע מפתח על מודל
שופט חזק יותר; (ג) להרחיב מהנתיב הגראונדד לנתיב האג׳נטי.

## 3. ניהול קונטקסט — precision מעל recall במה שמזינים ל־LLM

**קיים:** `SECTION_CHAR_LIMIT=2500` עם truncation גלוי (`agent/grounded.py`).

**חסר בהבנה:** התופעה של **lost-in-the-middle / context rot** — להזין יותר
chunks *מוריד* איכות, כי המחולל קובר את הרלוונטי. יותר הקשר ≠ תשובה טובה יותר.

**קשר ל־§1:** זו סיבה שנייה שבגללה reranker קריטי — הוא לא רק מדרג נכון,
הוא מאפשר להזין **פחות** chunks באיכות גבוהה במקום 8 בינוניים.

**הצעה:** אחרי reranker, לצמצם את מספר ה־sections שנכנסות לפרומפט (למשל
top-4 מדורגים במקום top-8 גולמיים) ולמדוד את ההשפעה על איכות התשובה (§2).

## 4. כיול הסירוב (refusal calibration)

**קיים:** `agent/guard.py` חוסם off-topic ו־injection דרך מרחק במרחב־אמבדינג.

**חסר:** המקרה ההפוך — שאלה **on-topic שה־retrieval פספס**. האם המחולל אומר
ביושר "לא מצאתי" ומפנה, או ממציא מ־chunk קרוב-אבל-לא-נכון? ה־hallucination
log מ־v0 היה על היעדר grounding; הסכנה בגרסה הגראונדד היא **over-trust
ב־chunk שגוי שאוחזר**.

**הצעה:** eval ייעודי עם שאלות שה־retrieval ידוע שנכשל עליהן — למדוד באיזה
אחוז המחולל מפנה ביושר לעומת ממציא בביטחון.

## 5. דפוסים אג'נטיים מעבר ללולאה

**קיים:** tool loop (`agent/loop.py`) + LangGraph twin, עם regeneration על
static-fail.

**חסר (מרכזי בשיח, לא חובה למוצר):**
- **Reflection / self-critique** — המחולל בודק את *איכות* תשובתו לפני החזרה,
  לא רק parse/symbols.
- **Self-consistency** — כמה מסלולים והצבעה על התשובה היציבה.

**הצעה:** לא קריטי לגרסה הנוכחית — לתעד כ־STRETCH מודע ולהעדיף §1–§2 קודם.

## 6. הצד התפעולי — ידוע בתוכנית, עוד לא נחווה

M4/M5 פתוחים: observability (Langfuse), caching, cost ceilings, rate-limit,
multi-tenancy. אלה לא "מושגי RAG" אבל הם ההבדל בין demo ל־product.

**הלקח שקשה ללמוד מקוד:** בלי trace לכל ריצה, כשמשהו נשבר בפרודקשן אתה עיוור —
לא תדע אם השבירה ב־tool, ב־retrieval, או במחולל. Langfuse (M4.1) הוא לכן לא
"nice to have" אלא תנאי לניפוי־שגיאות בכלל.

## סיכום — סדר עדיפויות מומלץ (מעודכן לפי המדידות של 2026-07-08)

1. **לסגור את מדידת הגלוסות** — הייצור רץ; אחריו re-embed (‏Build Index)
   ו־`eval-retrieval` מול ‏0.430/0.345. יחד עם תיקון ה־HNSW (#76) זה מטפל
   ישירות ברנקים שנמדדו (§1).
2. **Eval אוטומטי לאיכות תשובה** (LLM-judge/RAGAS) — בלעדיו לא יודעים אם
   שינוי שיפר או הרס (§2).
3. **Reranker כשלב precision** — אחרי שהגלוסות מכניסות את הדפים ל־pool;
   מדיד מייד מול `retrieval_v1` (§1, §3).
4. **HyDE / query transformation** — אם הגלוסות לא סוגרות את הפער לבדן (§1).
5. **צמצום context אחרי reranking** + מדידת ההשפעה (§3).
6. **Refusal eval** — כמה פעמים ממציא כשה־retrieval פספס (§4).
7. תפעול (§6) ודפוסים אג׳נטיים (§5) — לפי M4/M5, לא לפני 1–2.

**בשורה אחת:** הבסיס מעל הממוצע. הכשל המדוד הוא ברמת ייצוג האינדקס
(רנק 899–2,999) — ולכן הסדר: **גלוסות (רץ) → מדידה → reranker כ־precision
→ HyDE אם צריך**, ובמקביל **eval אוטומטי של איכות תשובה**.

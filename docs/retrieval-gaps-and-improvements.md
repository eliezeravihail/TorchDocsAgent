---
kind: gap-analysis
date: 2026-07-08
scope: retrieval quality + eval depth (RAG maturity review)
evidence: [eval/results/retrieval_v1.jsonl, eval/diagnose_retrieval.py, PLAN.md]
verdict: >
  Retrieval is the system's ceiling. The top gap is asymmetric dense
  retrieval on descriptive questions (recall=0 on c16/c17/c20) — a reranker
  is the single highest-leverage fix. Second gap: no automated answer-quality
  eval (only retrieval metrics + static checks).
open_gaps: [reranker, query-transformation, embedding-ceiling, answer-quality-eval, context-precision, refusal-calibration]
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

**עיקרון על:** ב־RAG ה־retrieval הוא התקרה. אם העמוד הנכון לא ב־top-k,
שום חוכמה של המחולל לא תשחזר אותו. הלולאה האג'נטית ממסכת חלק מזה עם ניסוח
מחדש — אבל לא יכולה לרנדר עמוד שאף שאילתה לא מביאה.

### שלושה מנופים לפתרון (מדורג לפי יחס תועלת/מאמץ)

| מנוף | מה עושה | מאמץ | מצב היום |
|---|---|---|---|
| **Reranker (cross-encoder)** | קורא query+doc **יחד**, לא שני וקטורים נפרדים — פותר ישירות את ה־asymmetry, מדרג מחדש top-20 | ~יומיים | STRETCH (PLAN §2.3) |
| **HyDE / query transformation** | LLM כותב תשובה משוערת, מאמבד **אותה** במקום השאלה — מקרב את ה־query לאזור של ה־doc | ~יום | לא קיים |
| **Embedding fine-tuned / גדול יותר** | bge-small הוא תקרה; asymmetry נפתר הכי טוב עם מודל שאומן על זוגות (query, doc) | ימים+CI | נבחר bge-small מודע ($0) |

**המלצה מיידית:** reranker על top-20. הוא המנוף הבודד הכי גבוה, והוא בר־מדידה
מייד מול `retrieval_v1` — אם c16/c17/c20 עוברים מ־recall 0 ל־1, זה מוכח במספר.

## 2. אין eval אוטומטי לאיכות התשובה (רק ל־retrieval)

**קיים:** recall/MRR ל־retrieval, ו־static checks (`eval/checks.py`: parse,
imports, symbols-in-index).

**חסר:** מדד אוטומטי ל**נכונות ונאמנות התשובה עצמה**. RAGAS מסומן STRETCH;
LLM-as-judge לא קיים.

**למה זה חשוב:** בלי faithfulness / answer-correctness אוטומטי, רגרסיות
מתגלות רק ידנית. זה הפער בין "פרויקט מרשים" ל"מערכת שאפשר לתחזק". **ה־eval
הוא המוצר.**

**הצעה:** LLM-judge קל על סט ה־40 שאלות (M4) — שלושה ציונים: faithfulness
(האם כל טענה נתמכת ב־context), answer-relevance, citation-correctness.
מריצים ב־Actions, שומרים ל־`eval/results/`, ומקבלים before/after לכל שינוי.

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

## סיכום — סדר עדיפויות מומלץ

1. **Reranker על top-20** — המנוף היחיד הכי גבוה; פותר ישירות את recall=0
   שכבר מדוד (§1). מדיד מייד מול `retrieval_v1`.
2. **Eval אוטומטי לאיכות תשובה** (LLM-judge/RAGAS) — בלעדיו לא יודעים אם
   שינוי שיפר או הרס (§2).
3. **HyDE / query transformation** — משלים את ה־reranker לשאלות התיאוריות (§1).
4. **צמצום context אחרי reranking** + מדידת ההשפעה (§3).
5. **Refusal eval** — כמה פעמים ממציא כשה־retrieval פספס (§4).
6. תפעול (§6) ודפוסים אג'נטיים (§5) — לפי M4/M5, לא לפני 1–2.

**בשורה אחת:** הבסיס מעל הממוצע. שני הדברים שבלעדיהם המערכת נשארת עם תקרה
מלאכותית הם **reranking (+HyDE)** ו־**eval אוטומטי של איכות תשובה**.

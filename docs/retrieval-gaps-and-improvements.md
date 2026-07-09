---
kind: gap-analysis
date: 2026-07-09
scope: retrieval quality + eval depth (RAG maturity review)
evidence: [eval/results/retrieval_v1.jsonl, eval/diagnose_retrieval.py,
  eval/results/agentic_v1_first6.jsonl, PLAN.md,
  "index-time enrichment survey (uploaded, 2026-07)"]
verdict: >
  The glosses × reranker combination broke the retrieval ceiling (measured
  2026-07-09): recall@8 0.430 → 0.460 (glosses) → 0.840 (reranker), MRR
  0.345 → 0.375 → 0.639, zero-recall 57 → 54 → 16. The two levers proved
  multiplicative — glosses put pages into the pool (largely via the tsvector
  channel), the cross-encoder realized them past the round-robin seat limit
  (39 questions flipped 0→1, one regressed). The 16-question residue splits
  into same-op sibling pages (an eval-pattern narrowness, not a retrieval
  failure), named-tutorial lookups crowded out by semantic look-alikes, and
  ~5 genuinely-buried pages — the QuOTE/HyPE hypothetical-question
  candidates. Judge baseline still pending.
open_gaps: [hypothetical-question-indexing, judge-baseline,
  eval-pattern-fidelity, context-precision, refusal-calibration]
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

### תוצאת הגלוסות — נמדד (2026-07-09, אחרי re-embed עם 3,632 גלוסות)

| גרסה | recall@8 | MRR | zero-recall |
|---|---|---|---|
| baseline | 0.430 | 0.345 | 57/100 |
| synopsis בלבד | 0.440 | 0.349 | 56/100 |
| **גלוסות (חי)** | **0.460** | **0.375** | **54/100** |

לפי סוג שאלה: **api 0.24, code 0.50, guide 0.80** — היעד של הגלוסות (api)
עדיין החוליה החלשה. וברמת הדף התמונה **בימודלית**:

| דף | רנק לפני | רנק אחרי | |
|---|---|---|---|
| LayerNorm | 899 | **2** | הגלוסה עבדה בדיוק כמובטח |
| torch.optim.SGD | 7 | **3** | השתפר |
| CrossEntropyLoss | 1,743 | 2,342 | גלוסה של משפט אחד לא הספיקה |
| torch.nn.Linear | 2,999 | 3,412 | כנ״ל |

**המסקנה המדודה:** הגלוסות הן מנוף אמיתי אבל לא אחיד — הן מכניסות חלק
מהדפים ל־pool (ושם reranker יממש אותם), ומשאירות אחרים קבורים (ושם צריך
העשרה עמוקה יותר — ראו §1.1).

### תוצאת ה־reranker — נמדד (2026-07-09, אותו אינדקס, ‏cross-encoder נוסף)

| | לפני reranker | **עם reranker** | שינוי |
|---|---|---|---|
| recall@8 | 0.460 | **0.840** | **+0.38** |
| MRR | 0.375 | **0.639** | **+0.26** |
| zero-recall | 54/100 | **16/100** | −38 |

לפי סוג: ‏api ‏0.24→**0.78**, ‏code ‏0.50→**1.00**, ‏guide ‏0.80→0.83.
‏39 שאלות התהפכו 0→1; רגרסיה אחת (g01, ראו למטה). ‏MRR עלה ב־46 וירד ב־9.

**הממצא המכניסטי:** שני המנופים הוכחו **מכפילים**, לא מחליפים. אפילו
CrossEntropyLoss — רנק dense ‏2,342 — התהפך ל־1.0: הגלוסה שלו ב־**tsvector**
תפסה את השאלה התיאורית בערוץ המילולי → הדף כן היה ב־pool → אבל חלוקת
המושבים הקבועה (round-robin, ‏~3 מושבים ל־kind) קברה אותו → ה־cross-encoder,
שקורא שאילתה+מועמד יחד על slate של 24, שלף אותו פנימה. "בעיית המושבים"
הייתה גדולה בהרבה משהעריך אבחון הרנקים הצפופים לבדו — הערוץ המילולי
המועשר הכניס ל־pool הרבה יותר דפים משחשבנו.

### ניתוח 16 שאלות ה־zero-recall הנותרות (2026-07-09)

השארית מתפרקת לארבעה סוגים שונים — עם תיקון שונה לכל אחד:

| סוג | שאלות | מה קורה | התיקון |
|---|---|---|---|
| **דף-אח של אותה פעולה** | a34 (gather), a35 (where), a43 (autograd.grad), a32 (stack) | נשלף `torch.Tensor.gather` במקום `torch.gather` (אותה פעולה, דף ה־method), או אח קרוב (concat/hstack במקום stack, backward במקום grad) | חלקו **צרות-תבנית של ה־eval** — להוסיף חלופות לגיטימיות ל־expected (‏Tensor.gather היא תשובה נכונה); חלקו דיוק סמלים ל־QuOTE |
| **תשובה טובה, תבנית צרה** | a40 (onnx), g12 (ddp) | נשלפו טוטוריאל ה־export ל־ONNX ודף ה־DDP API + סדרת multigpu — משתמש היה מרוצה; ה־gold דורש דף ספציפי | עדכון תבניות ה־eval (ביושר, בלי לרמות את המדד) |
| **חיפוש טוטוריאל בשם** | g01, g05, g25, g27 | "איפה הטוטוריאל של X" — עשרות דפים דומים סמנטית; הספציפי הנדרש נקבר בין דומיו. ‏g01 היא גם הרגרסיה היחידה: ‏FashionMNIST בשאלה גרר את דף ה־dataset של torchvision, וה־reranker העדיף אותו על ה־quickstart | מועמד ל־boost של kind=tutorial בשאלות "where's the tutorial", ו/או גלוסות לדפי tutorial (היום רק api) |
| **קבורים באמת** | a06 (Linear!), a07 (Conv2d), a27 (random_split), a28 (WeightedRandomSampler), a33 (einsum) | פרפרזה תיאורית שאף ערוץ לא גישר — ‏Linear עדיין ברנק ~3,400 גם אחרי גלוסה | **שאלות היפותטיות באינדוקס (QuOTE)** — אלה חמשת דפי היעד הראשונים |

**קריאה נכונה של ה־0.840:** recall אמיתי גבוה עוד יותר — סוגים 1–2 (6 שאלות)
הם ברובם הצלחות שהתבנית לא מזהה. הפער האמיתי שנשאר לעבודה אלגוריתמית
הוא סוגים 3–4 (‏~9–10 שאלות).

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

### המנופים לפתרון (מדורג מחדש לפי מדידת 2026-07-09 + סקירת הספרות)

| מנוף | מה עושה | מצב היום |
|---|---|---|
| **גלוסות קונטקסטואליות (Contextual Retrieval)** | משפט הקשר בשפה טבעית מוקדם לכל צ׳אנק לפני ההטבעה (וגם ל־tsvector) | **חי ונמדד: ‎+0.03 recall / ‎+0.03 MRR; בימודלי ברמת הדף** |
| **רוחב סריקת HNSW** | ראו "ממצא אינדקס" | **תוקן (#76)** |
| **Reranker (cross-encoder)** | קורא query+candidate יחד ומדרג מחדש slate רחב — שלב ה־**precision**; מממש את הדפים שהגלוסות הכניסו לרנק 2–20 | **חי ונמדד: ‏0.460→0.840 recall, ‏0.375→0.639 MRR — הקפיצה הגדולה של הפרויקט** |
| **שאלות היפותטיות באינדוקס (QuOTE/HyPE)** | ‏5–15 שאלות היפותטיות לכל דף, מאונדקסות לצדו — הופך את ההתאמה מ־question→document ל־**question→question**; המועמד לדפים שנשארו קבורים | הבא בתור; חמשת דפי היעד הראשונים זוהו (Linear, Conv2d, random_split, WeightedRandomSampler, einsum); תשתית הגלוסות ניתנת להרחבה ישירה |
| **HyDE / query transformation** | LLM כותב תשובה משוערת בזמן שאילתה ומאמבד אותה | נדחה לאחרי QuOTE — עלות latency+הזיה בכל שאילתה, לעומת עלות חד־פעמית באינדוקס |
| **Embedding גדול/מכוונן** | bge-base נוסה על v0 (סט קטן מדי — לא ראיה); התשתית להחלפה מדודה על v1 קיימת | פרמטרול מוזג (#70); קלף בקנה |

### 1.1 מה אומרת ספרות ההעשרה־בזמן־אינדוקס (סקירה, יולי 2026)

סקירת מחקר שנבחנה (2026-07-09) ממפה את המרחב לשתי משפחות משלימות —
הזרקת מונחים לצד ה־sparse (‏doc2query, SPLADE) והקרבת ייצוגים במרחב
האמבדינג לצד ה־dense (גלוסות, שאלות היפותטיות, propositions). הממצאים
שמכריעים אצלנו:

- **כלל ההחלטה** (המלצה 7 בסקירה): ‏Recall גבוה + דירוג נמוך → הבעיה
  ב־ranking → reranker; דף שכלל אינו מועמד → הבעיה בייצוג → העשרה.
  אצלנו שני המקרים בו־זמנית — ולכן שני המנופים למעלה, בסדר הזה.
- **Reranking קריטי למימוש** (‏Anthropic: ‏49%→67% הפחתת כישלונות בתוספת
  reranker; ‏"Reconstructing Context"‏, arXiv:2504.19754: ה־reranking היה
  קריטי למימוש הפוטנציאל). המספרים של Anthropic הם הערכה פנימית, לא
  peer-reviewed — לצפות לפחות (וכך אכן מדדנו בגלוסות).
- **שאלות היפותטיות באינדוקס** (‏QuOTE, ‏arXiv:2502.10976; ‏HyPE) — אותה
  תובנה של HyDE אבל בעלות חד־פעמית באינדוקס במקום latency+הזיה בכל
  שאילתה; ‏10–15 שאלות לצ׳אנק, מחיר באחסון (×5 אמבדינגים). אצלנו: רק
  לדפי api קבורים.
- **אזהרת Weller et al.‏** (‏Findings of EACL 2024, ‏arXiv:2309.08541, על
  11 טכניקות/12 דאטהסטים/24 מודלים): הרחבה גנרטיבית מועילה למאחזרים
  חלשים ומזיקה לחזקים. ‏bge-small לא מכוונן = חלש = ההעשרה אצלנו צפויה
  לעזור — עקבי עם המדידה.
- **לא** doc2query גולמי בלי סינון (‏Doc2Query--‏, ECIR 2023: הזיות מנפחות
  את האינדקס ופוגעות ביעילות); ‏SPLADE דורש אימון GPU; ‏GraphRAG/RAPTOR
  פותרים שאלות גלובליות — לא הכאב הנמדד שלנו.

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

## סיכום — סדר עדיפויות מומלץ (מעודכן לפי המדידות של 2026-07-09, אחה"צ)

1. ~~מדידת הגלוסות~~ — **בוצע**: ‏0.430→0.460 recall (§1).
2. ~~מדידת ה־reranker~~ — **בוצע**: ‏0.460→**0.840** recall, ‏0.375→**0.639**
   MRR, ‏zero-recall ‏54→16. שני המנופים מכפילים (§1).
3. **baseline ל־judge** — ה־retrieval השתנה דרמטית; ריצת `suite=judge`
   ראשונה נותנת את עוגן האיכות להמשך (§2).
4. **שאלות היפותטיות באינדוקס (QuOTE-style)** לחמשת דפי היעד שזוהו
   (Linear, Conv2d, random_split, WeightedRandomSampler, einsum) —
   הרחבת pipeline הגלוסות (§1.1, ניתוח ה־16).
5. **נאמנות תבניות ה־eval** — להוסיף חלופות לגיטימיות ל־expected
   (‏Tensor.gather, טוטוריאל ה־ONNX...) כדי שהמדד ישקף הצלחות אמיתיות;
   בזהירות וביושר, לא כדי לנפח מספר (ניתוח ה־16).
6. **ניסוי k=4** — ‏precision-over-volume אחרי ה־reranker; כפתור `k` קיים
   ב־workflow (§3). למדוד עם ה־judge, לא רק עם recall.
7. **Refusal eval** (§4), תפעול (§6) ודפוסים אג׳נטיים (§5) — לפי M4/M5.

**בשורה אחת:** ‏0.430 → 0.460 → **0.840** recall (‏MRR ‏0.345 → 0.639) —
גלוסות ל־candidate generation ו־cross-encoder ל־precision, מכפילים זה את
זה. השארית: ‏QuOTE לחמשת הקבורים, תבניות eval כנות יותר, ועוגן judge.

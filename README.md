# Metadata Extraction from Rental Agreements

An AI system that reads a rental agreement, either a **scanned image (.png)** or a **Word file (.docx)**, and extracts six fields, whatever the template:

| Field | Meaning |
|---|---|
| Aggrement Value | monthly rent (digits only) |
| Aggrement Start Date | `DD.MM.YYYY` |
| Aggrement End Date | `DD.MM.YYYY` |
| Renewal Notice (Days) | notice period in days |
| Party One | landlord / lessor |
| Party Two | tenant / lessee |

(The column spelling "Aggrement" follows the dataset's CSV headers.)

**No rule-based extraction is used.** There is no RegEx and there are no hand-written conditions that find fields. Every field is produced by a language model. The only non-model code is file reading (python-docx, OCR), cutting the JSON object out of the model's reply, and string comparison for scoring.

---

## Approach

```
.docx  --python-docx-->  plain text --+
                                      +--> LLM + few-shot examples --> JSON (6 fields)
.png   --EasyOCR------>  plain text --+
```

1. **Document to text.** `.docx` files are read with `python-docx` (paragraphs and tables). Scans are read with EasyOCR. After this step the model does not care which format the document came from.
2. **Text to fields.** A language model gets a system prompt (output keys, date format, how to treat names, notice periods and end dates), two worked examples from the training set (few-shot), and the document text. It answers with a JSON object. Decoding is deterministic (`temperature 0` / greedy).
3. **Evaluation.** Exact-match recall per field, as defined in the assignment.

### Why few-shot prompting and not fine-tuning
- The training set has only 10 labelled documents, which is far too few to fine-tune a model without overfitting. They work much better as prompt examples and as a validation set.
### Two models, same pipeline
| Option | Model | Needs |
|---|---|---|
| A (fully offline) | `Qwen/Qwen2.5-7B-Instruct`, 4-bit | NVIDIA GPU (Kaggle T4 x2 works), free |
| B (best result) | `gemini-3.8-flash` | Google AI Studio API key (free tier) |

> **Privacy note:** on Gemini's free tier, prompts may be used by Google to improve their products. Option B sends the document text to Google. Option A keeps everything local.

---

## Results

Exact-match recall per field. **Train** = leave-one-out on the 9 usable training documents (the document being tested is never one of its own examples). **Test** = the 4 documents in `test/`.

### Test set (4 documents; each document is worth 25 points per field)

| Run | Value | Start | End | Renewal | Party One | Party Two | Avg |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (final prompt) | 0.75 | 1.00 | 0.75 | 0.75 | 1.00 | 0.25 | 0.75 |
| Gemini 3.8 Flash | 0.75 | 0.75 | 0.75 | 1.00 | 1.00 | 0.25 | 0.75 |
| *Qwen2.5-7B, first run (see note)* | 0.75 | 1.00 | 0.75 | 0.75 | 0.50 | 0.00 | 0.63 |

### Train set (9 documents, leave-one-out)

| Run | Value | Start | End | Renewal | Party One | Party Two | Avg |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.89 | 0.78 | 0.44 | 0.78 | 0.11 | 0.44 | 0.57 |
| Gemini 3.8 Flash | 0.89 | 0.78 | 0.44 | 0.78 | 0.33 | 0.67 | 0.65 |

### How to read these numbers honestly
- **The test set is tiny.** With 4 documents, one document moves a field by 25 points. The two test averages tie, and the train set (more documents) favours Gemini, mainly on party names.
- **The prompt was revised once after looking at test errors.** The first Qwen test run (third row) used an earlier prompt. Its errors showed titles such as "Sri" and "Mrs." being kept in party names, so I strengthened the name rule and the end-date rule (the examples in the prompt are invented, not taken from test documents). The "final prompt" rows come after that revision, so they are not a fully clean held-out score. The first-run row is the cleaner one.
- **Several misses cannot be fixed by any model** because the labels disagree with the documents (see below).

### Data quirks found along the way
- **Two training labels are swapped.** The CSV rows for `54770958-Rental-Agreement` and `54945838-Rental-Agreement` belong to each other's files. The first file's text contains Sadasivuni and the second's contains Parthasarathy, but the CSV says the opposite. I confirmed this by checking the text of each file, and swapped the two rows **in memory only** for scoring and for few-shot examples. The CSV files are not modified, and no test labels are touched.
- **`24158401-Rental-Agreement` has a row in `train.csv` but its file is in `test/`.** It is excluded from training, from the examples and from train scoring.
- **`46239065-Standard-Rental-Agreement-...docx` is in `train/` but has no row** (it is a US agreement), so it is ignored.
- **Some labelled end dates are not valid dates** (`31.11.2009`, `31.04.2011`, `31.02.2011`) and some labels contain typos relative to the document (`GERALDINE Q.` where the text reads `O.`, `Ingles,` vs `Ingles.`, `.B.Kishore`). Dates are therefore kept as plain text and never run through a date parser, and exact-match scores on these cannot reach 100%.
- Test files named `*.pdf.docx` are matched to the CSV by removing the extra `.pdf`.

## Reproducing the predictions

### On Kaggle (what I used)
1. Upload `data/` as a Kaggle Dataset and add it to a new notebook.
2. Notebook settings: **Accelerator = GPU T4 x2** (needed for Option A, and for fast OCR) and **Internet = On**.
3. Open `notebook.ipynb` and set `DATA = "/kaggle/input/<your-dataset>/data"`.
4. For Option B, add a Kaggle secret named `GEMINI_API_KEY` (Add-ons, Secrets) and make sure it is attached to the notebook.
5. Run the notebook top to bottom. Option A and Option B are separate sections, and either can be run alone.

### Locally
```bash
pip install -r requirements.txt        # uncomment the Qwen lines if you want Option A
export GEMINI_API_KEY=your_key_here    # Option B only
jupyter notebook notebook.ipynb        # set DATA to your data folder
```
The Qwen model (about 15 GB) downloads automatically from Hugging Face on first run.

### Notes for re-running
- Outputs are deterministic, but a hosted model can change over time, so a later Gemini run may differ slightly.
- The free Gemini tier has rate limits and can return temporary 503/429 errors. The notebook retries and pauses between calls. Do not allow fallback to a different model mid-run, or the scores will mix models.

---

## REST API

`app.py` wraps the system as a web service. One endpoint takes a file and returns the six fields.

### Start the server
```bash
export GEMINI_API_KEY=your_key_here
uvicorn app:app --host 0.0.0.0 --port 8000
```
- Uses Gemini by default. To use the local model instead: `BACKEND=qwen uvicorn app:app ...` (needs a GPU).
- Optional settings: `GEMINI_MODEL`, `QWEN_MODEL`, `SHOTS_PATH`.
- `shots.json` holds the two few-shot examples. Export it from the notebook once (cell below). Without it the API still runs, but zero-shot, which scores lower.

```python
# run in the notebook after make_shots is defined
import json
json.dump(make_shots(None), open("shots.json", "w", encoding="utf-8"), ensure_ascii=False)
```

### Call it
```bash
curl -X POST http://localhost:8000/extract -F "file=@agreement.docx"
```

```python
import requests
r = requests.post("http://localhost:8000/extract",
                  files={"file": open("scan.png", "rb")})
print(r.json())
```

Response:
```json
{
  "file_name": "agreement.docx",
  "metadata": {
    "Aggrement Value": "4000",
    "Aggrement Start Date": "05.12.2008",
    "Aggrement End Date": "31.10.2009",
    "Renewal Notice (Days)": "90",
    "Party One": "K.Kuttan",
    "Party Two": "P.M. Narayana Namboodri"
  }
}
```
(Illustrative values. A field the model cannot find is returned as `null`.)

| Endpoint | Purpose |
|---|---|
| `POST /extract` | multipart upload, field name `file`; accepts `.docx`, `.png`, `.jpg`, `.jpeg` |
| `GET /health` | shows the active backend, model and number of few-shot examples |
| `GET /docs` | interactive Swagger page (built in) |

Errors: `400` unsupported file type, `422` unreadable or empty document, `502` model call failed.

---

## Limitations
- Only 4 test documents, so scores are noisy; train scores come from only 9 documents, some with unmatchable labels.
- Scanned documents depend on OCR quality. Broken spacing or merged letters from the scan carry through to the extracted names.
- End dates are often labelled by a convention that is not written in the document (start date plus period, minus one day). The prompt teaches this convention, but the model's date arithmetic is not always right.
- Written for rental agreements in the style of the provided dataset (English, mostly Indian templates).

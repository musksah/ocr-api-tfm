# ocr-api-tfm

A FastAPI service that reads two photos of a Spanish-language food package,
one of the ingredients panel and one of the nutrition facts table. It extracts
the ingredient list and the total sugar value with OCR, translates the
ingredients to English, and forwards both to a separate prediction service.
That service returns a NOVA food-processing group and a diabetes-impact label.

This is the image-processing service of a master's thesis (TFM) prototype.
The prediction model it calls is **not** part of this repository. The code was
last changed in September 2024 and should be treated as a research prototype,
not a production service. See [Limitations](#limitations).

> The output of this project is not medical or nutritional advice.

## How it works

```text
POST /process_images/  (files[0] = ingredients photo, files[1] = nutrition facts photo)
        │
        ├── files[1] → grayscale (OpenCV) → PaddleOCR (lang="en")
        │               → find the text box "azucares totales"
        │               → take the leftmost box on the same row (±50 px) → float
        │
        ├── files[0] → grayscale + 3×3 erosion (OpenCV) → PaddleOCR (lang="es")
        │               → keep text from the first line containing "ingredientes" onwards
        │               → split on , : ( ) .  → drop fragments with digits or ≤1 char
        │               → NLTK POS filter: drop phrases with a verb, keep phrases with a noun
        │               → Spanish → English via Google Cloud Translation (cached in translations.json)
        │
        └── {"ingredientlist": [...], "sugar": <float>}
                → POST to the prediction service (URL hardcoded in main.py)
                → {"nova": ..., "diabetes_impact": ...}
```

`ingredientlist` contains both the Spanish phrases and their English
translations, deduplicated with a `set`, so its order is not stable.

## Repository layout

| Path | What it is |
|---|---|
| `main.py` | The whole API: the endpoint, OCR preprocessing, text filtering, translation cache, sugar parsing |
| `utils.py` | `ModelInputGenerator`: turns an ingredient list into a count vector over the vocabulary in `ingredients.pkl`. **Not used by `main.py`** since v1.1, which replaced local vectorisation with a call to the prediction service. |
| `ingredients.pkl` | A pickled Python list of 19,529 strings: 19,526 ingredient names (multilingual and noisy), followed by the column names `nova_group`, `product_name`, `_id` |
| `translations.json` | Spanish → English translation cache. The API reads it at startup and **rewrites it on every request**. |
| `cereal.jpg`, `temp_image.jpg` | Sample photos of a cereal box's nutrition and ingredients panel |
| `bw_temp_image.jpg` | Grayscale version of `temp_image.jpg`, the kind of intermediate file the API writes while processing |
| `requirements.txt` | Direct dependencies, pinned to the versions used during development |

## Requirements

- Python 3.9 (the version used during development)
- About 1 GB of disk for the virtual environment (PaddlePaddle, OpenCV,
  spaCy). On first use PaddleOCR also downloads its detection,
  recognition and angle-classification models to `~/.paddleocr/`.
- Network access at startup, because `main.py` calls `nltk.download(...)` at import time
- Google Cloud credentials with the Cloud Translation API enabled (see below)
- A running prediction service (see [Prediction service contract](#prediction-service-contract))

## Setup

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.9 app-env
uv pip install --python app-env -r requirements.txt
```

Or with plain `pip`:

```bash
python3.9 -m venv app-env
source app-env/bin/activate    # Windows: app-env\Scripts\activate
pip install -r requirements.txt
```

`app-env/` is already in `.gitignore`.

`requirements.txt` pins only the direct dependencies, so transitive
versions are chosen at install time. `paddlepaddle` is pinned to 2.6.2
because 2.6.1, the version used originally, is no longer on PyPI.

If `paddlepaddle` fails to install on your platform, see the
[PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/en/install/quick).
`paddlepaddle-gpu` can replace `paddlepaddle` on a machine with CUDA.

### Google Cloud Translation

`main.py` uses `google.cloud.translate_v2.Client()`, which picks up
[Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials):

```bash
gcloud auth application-default login
# or
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Every phrase that is not already in `translations.json` is sent to the
Translation API, which is billed per character above the free tier.

### Prediction service

The prediction service URL is hardcoded in `main.py` as `url = "https://….ngrok-free.app/predict"`.
That was a temporary ngrok tunnel used during development and no longer
resolves. Point it at your own service before calling the endpoint.

## Running

```bash
uvicorn main:app --reload
```

Interactive docs are at <http://localhost:8000/docs>.

## Usage

The two files are read by position, so the order matters: ingredients
first, nutrition facts second.

```bash
curl -X POST http://localhost:8000/process_images/ \
  -F "files=@ingredients.jpg" \
  -F "files=@nutrition_facts.jpg"
```

The sample `cereal.jpg` shows both panels in one photo, so you can send it
twice:

```bash
curl -X POST http://localhost:8000/process_images/ \
  -F "files=@cereal.jpg;filename=ingredients.jpg" \
  -F "files=@cereal.jpg;filename=nutrition.jpg"
```

Use different `filename`s for the two parts. The API saves uploads under
their client-supplied names, so two files with the same name overwrite each
other.

On success the response is whatever the prediction service returned for
these two keys:

```json
{ "nova": 3, "diabetes_impact": "Medio" }
```

### Prediction service contract

The API sends:

```json
{ "ingredientlist": ["maiz", "corn", "azucar", "sugar", "sal", "salt"], "sugar": 9.4 }
```

It expects HTTP 200 with a JSON object containing `nova` and
`diabetes_impact`. Any other status is meant to produce HTTP 500. See
[Limitations](#limitations) for why it does not do that cleanly.

## Limitations

This is a thesis prototype and has the rough edges of one. These are
known problems in the current code:

**Correctness**

- On a non-200 response from the prediction service, the error branch
  references an undefined variable `e` and raises `NameError`. It also
  imports `HTTPException` from `http.client` instead of FastAPI. The client
  gets a generic 500 instead of the intended message.
- Sugar extraction matches the literal lowercase string `azucares totales`
  **without** an accent. It relies on the English PaddleOCR model dropping
  the accent from "Azúcares". If the OCR output keeps it, no sugar value is
  found.
- The sugar value is parsed by removing `g` and calling `float()`. A comma
  decimal (`9,4 g`), a different unit, or a missing row all become `0.0`,
  which is indistinguishable from a real zero.
- The "same row" test uses a fixed 50-pixel window, so it depends on image
  resolution and framing. With labels that have "per 100 g" and "per
  serving" columns, the leftmost value (usually per 100 g) wins.
- Ingredient extraction starts at the first line containing `ingredientes`
  and has no end marker. Later text on the package with no digits leaks in.
  On the sample photos the output includes `kellogg company`,
  `battle creek` and `michigan`.
- The POS filter runs NLTK's **English** tagger and English WordNet
  lemmatizer on **Spanish** text. It is not a meaningful filter for
  Spanish. On the samples it drops `malta en polvo`, and the digit filter
  drops every `vitamina …` entry.
- PaddleOCR can return `[None]` for an image with no detected text, and
  `cv2.imread` returns `None` for a non-image upload. Neither is handled,
  so both give a 500.
- Fewer than two uploaded files raises `IndexError`, which also gives a 500.

**Operational and security**

- Uploads are written to the working directory under the client-supplied
  filename, which allows path traversal (`../`) and collisions between
  concurrent requests. Temporary files are not deleted if processing fails.
- `translations.json` is rewritten on every request without locking, and
  the translation client is re-created on every call.
- Two PaddleOCR models are created on **every** request. On the first
  request they are also downloaded, which took about two minutes in testing.
- The prediction service URL is hardcoded and points to an expired tunnel.
  There are no environment variables or configuration file.

**Project hygiene**

- No tests and no CI yet.
- `utils.py`, `ingredients.pkl` and the spaCy-based `filtrar_textos()` are
  unused. `filtrar_textos()` would also need the `es_core_news_sm` model,
  which is not in `requirements.txt`.
- `spacy` is installed only because `main.py` imports it at module level.

## What has been verified

Checked on Windows 11 with Python 3.9 in October 2026, without Google
Cloud and without the prediction service:

- `requirements.txt` resolves for Windows, Linux and macOS, and installs
  and imports on Windows.
- Running the OCR steps on the two sample photos gives `sugar = 9.4` for
  both, and an ingredient list with the issues described above.
- With the prediction service stubbed, the non-200 path raises `NameError`,
  and a request with a single file raises `IndexError`.

Not verified: the Translation API call, a real prediction service, and
installation on Linux or macOS. The script used is
`.claude/skills/ocr-smoke-test/smoke.py`.

## Data and third-party content

- The sample photos show the packaging of a commercial cereal product. The
  packaging text and branding belong to their owner. They are included only
  as OCR test inputs.
- `ingredients.pkl` is a plain list of strings with no executable content.
  Still, only load pickle files you trust. Its trailing column names
  (`_id`, `product_name`, `nova_group`) suggest it was built from
  [Open Food Facts](https://world.openfoodfacts.org/) data. This origin is
  not confirmed. Open Food Facts data is licensed under the
  [ODbL](https://opendatacommons.org/licenses/odbl/1-0/), which requires
  attribution, and the MIT license below does not cover it.

## License

The code is released under the [MIT License](LICENSE). Third-party content
described above (product packaging in the sample photos, and any Open Food
Facts data) keeps its own terms.

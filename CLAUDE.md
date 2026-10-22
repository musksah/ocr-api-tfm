# CLAUDE.md

FastAPI service from a master's thesis (TFM) prototype. It OCRs two photos of
a Spanish food label (ingredients panel, nutrition facts panel) with
PaddleOCR, extracts ingredient phrases and total sugar, translates the
ingredients es→en with Google Cloud Translation, and POSTs the result to an
external prediction service that returns `nova` and `diabetes_impact`. That
service is not in this repo. The repo is public.

## Layout

- `main.py`: everything. Module level runs six `nltk.download(...)` calls,
  loads `translations.json` from the CWD and hardcodes the predictor `url`.
  One endpoint: `POST /process_images/` (`files: list[UploadFile]`,
  `files[0]` = ingredients, `files[1]` = nutrition facts).
  - `get_sugar_percentage()`: grayscale → PaddleOCR `en` →
    `extract_total_sugar()` (row match on `"azucares totales"`, ±50 px in y,
    leftmost box) → `get_sugar_value()` (strip `g`, `float`, else `0.0`).
  - `get_ingredients()`: grayscale + 3×3 erode → PaddleOCR `es` →
    `preprocess_text()` (start at first line with `ingredientes`, split on
    `[,:().]`, drop digits and short fragments) → `filtrar_textos_nltk()`
    (English NLTK POS tagger: drop phrases with verbs, keep phrases with
    nouns, lemmatize nouns) → `translate_ingredients()`.
  - `translate_text()`: cache lookup in `translations.json`, Translation API
    for misses, then rewrites the whole JSON file.
  - `filtrar_textos()` (spaCy `es_core_news_sm`) is dead code.
- `utils.py`: `ModelInputGenerator`, a count vector over `ingredients.pkl`.
  Unused since v1.1.
- `ingredients.pkl`: pickled `list[str]`, 19,526 vocabulary strings followed
  by `nova_group`, `product_name`, `_id`. Inspect it with `pickletools`, not
  `pickle.load`, if you only need to look.
- `cereal.jpg`, `temp_image.jpg`, `bw_temp_image.jpg`: sample label photos.

## Commands

```bash
uv venv --python 3.9 app-env
uv pip install --python app-env -r requirements.txt
uvicorn main:app --reload          # run from the repo root (relative paths)
```

`requirements.txt` lists the direct dependencies only. The original
`pip freeze` was UTF-16, had unsatisfiable pins (`google-cloud-core 1.7.3`
needs `google-auth<2`, `grpcio-status 1.66.1` needs `protobuf>=5`) and
pinned `paddlepaddle==2.6.1`, which is no longer on PyPI. Keep `protobuf`
at 3.20.x because PaddlePaddle 2.6 needs it.

There are no tests or CI. Use the `ocr-smoke-test` skill to check a change.

## Environment

No env vars are read by the code. External dependencies:

- Google Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS`
  or `gcloud auth application-default login`) for Cloud Translation. This is
  a paid API. Never call it without asking the user which project to use,
  and never use their production projects.
- The predictor `url` in `main.py` is an expired ngrok tunnel. Ask the user
  for a real endpoint, or stub `requests.post`.

## Known issues (in the code, not yet fixed)

- Non-200 branch: `HTTPException` is imported from `http.client`, and `e`
  is undefined. Both are wrong. Use `fastapi.HTTPException`.
- `translate_text` has `text = text.decode(...)` for bytes input, so it uses
  `text` before assigning it.
- Uploads are saved as `./{client filename}`: path traversal, collisions
  between requests, and leftover files when processing raises.
- Two `PaddleOCR(...)` models are built on every request. Hoist them to
  module level.
- `result[0]` can be `None` (no text found), `cv2.imread` can return `None`,
  and fewer than two files gives `IndexError`. All surface as 500.
- Sugar parsing returns `0.0` for "not found" and "unparseable" alike, and
  `9,4 g` (comma decimal) parses as `0.0`.
- `"azucares totales"` is matched without an accent. It works only because
  the `en` OCR model tends to drop the accent.
- `translations.json` is rewritten on every request without locking. Don't
  commit noisy cache churn from local runs.

## Conventions

- Comments and many identifiers are in Spanish (`filtrar_textos`,
  `imagen_erosionada`), and the README and these docs are in English. Keep
  the existing mix when editing a function, and don't mass-rename.
- Keep the single-file structure unless the user asks for a refactor.
- Sample images are product packaging photos. Don't add new photos that
  show people, documents, addresses or screens.
- Before committing, run the `public-repo-check` skill.

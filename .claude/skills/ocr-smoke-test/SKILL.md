---
name: ocr-smoke-test
description: Check the OCR pipeline offline, without Google Cloud Translation or the prediction service, by running the sugar and ingredient extraction on the sample label photos and exercising the endpoint's error paths. Use after changing main.py, the OCR preprocessing, the text filters, or requirements.txt, or when the user asks to "probar el OCR".
---

# OCR smoke test

There are no unit tests. `smoke.py` in this folder is the regression check.

## What it does

- Sets `GOOGLE_APPLICATION_CREDENTIALS` to a nonexistent file, so an
  accidental Translation API call fails instead of using the user's ADC.
  The user's default GCP project may be production.
- Replaces `main.translate_ingredients` with a lookup in `translations.json`
  only. Misses print `<MISS>`.
- Runs `get_sugar_percentage` and `get_ingredients` on `cereal.jpg` and
  `temp_image.jpg`.
- Stubs `requests.post` to return 503, then calls the endpoint with two
  files and with one file, and prints the exceptions.

## Run it

Run on a **copy** of the repo. `main.py` writes upload copies into the CWD
and can rewrite `translations.json`.

```bash
uv venv --python 3.9 app-env && uv pip install --python app-env -r requirements.txt
RUN=$(mktemp -d)
cp main.py utils.py translations.json ingredients.pkl cereal.jpg temp_image.jpg "$RUN"/
(cd "$RUN" && PYTHONIOENCODING=utf-8 "$OLDPWD"/app-env/bin/python "$OLDPWD"/.claude/skills/ocr-smoke-test/smoke.py)
```

On Windows the interpreter is `app-env\Scripts\python.exe`. The first run
downloads the PaddleOCR models (about two minutes) and the NLTK data. Run it
in the background.

## Baseline (October 2026, before any code fixes)

- `sugar=9.4` for both images. The `en` OCR returns `Azucares totales`
  without an accent, which is why the match works.
- Ingredients start with `maiz`, `min`, `azücar`/`azucar`, `sal`, … and
  wrongly end with manufacturer text (`kellogg company`, `battle creek`,
  `michigan`). `malta en polvo` and the `vitamina …` entries are missing.
- Non-200 path: `NameError: name 'e' is not defined`. One file: `IndexError`.
- `leftover files` lists only the two input images.

Report how the output differs from this baseline. If a change fixes one of
the known issues, update this baseline and the README's Limitations section.

## Not covered

The real Translation API (billed, so ask the user which project to use
first) and a real prediction service (ask for its URL). Never point either
at production without the user's explicit OK.

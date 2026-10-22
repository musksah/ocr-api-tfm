---
name: public-repo-check
description: Scan staged changes, or the whole tracked tree and history, for secrets, credentials, personal data, private datasets and university or third-party material before committing or pushing to this public repo. Use before any commit that adds files, images or data, or when the user asks whether the repo is safe to share.
argument-hint: [--all]
---

# Public repo check

This repo is public. Anything committed stays in git history even after a
later commit deletes it.

Scope: staged changes (`git diff --cached --stat` and `git diff --cached`) by
default. With `--all`, check `git ls-files`, and search history with
`git log -p --all -S '<pattern>'` and
`git grep -I -n -E '<pattern>' $(git rev-list --all)`.

## Look for

- **Credentials:** service-account JSON (`"type": "service_account"`,
  `private_key`), Google API keys (`AIza...`), `.env` files, tokens in URLs,
  ngrok authtokens. `main.py` uses ADC, so no key should ever be in the code.
- **Endpoints that identify people:** ngrok URLs embed the public IP of the
  machine running the tunnel (the `url` in `main.py` already does, and it is
  in history). Prefer an env var or a placeholder for any new endpoint.
- **Images:** this repo stores label photos. For each new `*.jpg`, `*.png`
  or `*.heic`:
  - Look at it. Reject it if it shows faces, documents, screens, addresses or
    identifiable home interiors.
  - Check EXIF for GPS, device and time. Never `pickle.load` anything to do
    this, and run Pillow with `python -I`:
    `Image.open(p).getexif()` and `.get_ifd(0x8825)` for GPS.
  - Strip metadata before committing if it has any.
  - Leftover `bw_*.jpg` or upload copies in the repo root come from a request
    that crashed before cleanup. Don't commit them.
- **Data files:** new `*.pkl`, `*.csv`, `*.parquet`, `*.npy` or model weights.
  Ask where the data comes from and whether its license allows
  redistribution. Open Food Facts data is ODbL and needs attribution. Inspect
  pickles with `pickletools.genops` (look for `GLOBAL`/`STACK_GLOBAL`
  opcodes) instead of loading them.
- **Cache churn:** `translations.json` is rewritten by every request. Commit
  only intentional additions, and check that it holds label text only, with
  no user-submitted text that could be personal.
- **University and team material:** thesis chapters, grading rubrics, course
  datasets, or code written by other TFM team members. Ask the user before
  adding any of it, and before changing the license.
- **Personal data:** emails, phone numbers and Colombian ID numbers (cédula,
  NIT) in fixtures, logs or sample data. Commit author emails are already
  public in history, so don't add more.

## Report

List findings by severity with file and line (or commit and path). Show only
the first 4 characters of any secret. For a real credential, tell the user to
rotate or revoke it first, because removing it from the repo is not enough.
Never rewrite history or force-push yourself. If history needs cleaning,
explain `git filter-repo` and let the user run it.

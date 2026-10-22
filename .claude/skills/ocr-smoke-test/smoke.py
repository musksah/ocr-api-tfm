"""Offline smoke test for ocr-api-tfm. Run with cwd = a COPY of the repo.

No GCP: translate_ingredients is replaced by a cache-only lookup and ADC is
pointed at a nonexistent file. No predictor: requests.post is stubbed.
"""
import asyncio, io, os, sys, time, json
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("does-not-exist.json")
sys.path.insert(0, os.getcwd())

t0 = time.time()
import main  # noqa: E402  (runs nltk.download at import)
print(f"import main: {time.time()-t0:.1f}s")

from starlette.datastructures import UploadFile  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402

async def cache_only(texts):
    return [(t, main.translations.get(t, "<MISS>")) for t in texts]
main.translate_ingredients = cache_only

def up(path, name):
    return UploadFile(file=io.BytesIO(open(path, "rb").read()), filename=name)

async def run():
    t = time.time()
    es = PaddleOCR(use_angle_cls=True, lang="es", show_log=False)
    en = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    print(f"model init: {time.time()-t:.1f}s")
    for img in ["cereal.jpg", "temp_image.jpg"]:
        t = time.time()
        sugar = await main.get_sugar_percentage(up(img, "n_" + img), en)
        t1 = time.time() - t; t = time.time()
        ingr = await main.get_ingredients(up(img, "i_" + img), es)
        print(f"== {img}: sugar={sugar!r} ({t1:.1f}s) ingredients ({time.time()-t:.1f}s):")
        for pair in ingr:
            print("   ", pair)
    # Raw OCR of the sugar row to see whether the accent is dropped
    res = en.ocr("cereal.jpg")[0]
    print("en OCR lines containing 'zucar':", [r[1][0] for r in res if "zucar" in r[1][0].lower()])
    res = es.ocr("cereal.jpg")[0]
    print("es OCR lines containing 'zucar':", [r[1][0] for r in res if "zucar" in r[1][0].lower()])

    # Endpoint error path: predictor returns 503
    class R:  status_code = 503
    main.requests.post = lambda *a, **k: R()
    try:
        await main.create_upload_files([up("cereal.jpg", "a.jpg"), up("cereal.jpg", "b.jpg")])
    except Exception as e:
        print("non-200 path raised:", type(e).__module__, type(e).__name__, e)
    try:
        await main.create_upload_files([up("cereal.jpg", "c.jpg")])
    except Exception as e:
        print("one-file path raised:", type(e).__name__, e)

asyncio.run(run())
print("leftover files:", sorted(f for f in os.listdir(".") if f.endswith(".jpg")))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from paddleocr import PaddleOCR
from fastapi.responses import JSONResponse
import os, re, cv2
from rapidfuzz import fuzz
from rapidfuzz import process

app = FastAPI()

# Diccionario de ingredientes
ingredient_dict = ['Leche semidescremada', 'sal', 'harina', 'huevo', 'leche', 'Cloruro de calcio', 'azucar', 'malta', 'maiz']

@app.post("/process_images/")
async def create_upload_files(files: list[UploadFile] = File(...)):

    img_ingredients = files[0]

    img_path = f"./{img_ingredients.filename}"

    # Guardar archivo
    with open(img_path, "wb") as f:
         f.write(await img_ingredients.read())

    # Leer la imagen con OpenCV
    img = cv2.imread(img_path)

    # Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Guardar la imagen en blanco y negro
    bw_img_path = f"./bw_{img_ingredients.filename}"
    cv2.imwrite(bw_img_path, gray_img)
    # Inicializar el modelo OCR
    ocr_model = PaddleOCR(lang='en')

    # Obtener el texto de la imagen
    result = ocr_model.ocr(bw_img_path)
    print("Result: ")

    text_list = arrange_text(result[0])

    result = search_ingredients(text_list)
    result_example = {
        "nova": 3,
        "diabetes_risk": "Medio"
    }
    return JSONResponse(content=result_example)


def search_ingredients(data):
    matches = set()
    for word in data:
        if word in ingredient_dict:
            matches.add(word)
        else:
            match = process.extractOne(word, ingredient_dict, scorer=fuzz.ratio)
            if match and match[1] >= 60:
                matches.add(match[0])
    return list(matches)


def arrange_text(data):
    # Array de resultado
    result_texts = []

    hasIngredients = False
    # Iterar sobre cada elemento de la lista
    for item in data:
        #Acceder al texto
        text = item[1][0]
        if(hasIngredients):
            separated_texts = re.split(r'[,:().]', text.lower())
            result_texts.extend(separated_texts)
        else:
            hasIngredients = "ingredientes" in text.lower() 
            if hasIngredients:
                separated_texts = re.split(r'[,:().]', text.lower())
                result_texts.extend(separated_texts)
    return result_texts
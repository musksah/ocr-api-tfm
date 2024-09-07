from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import HTMLResponse
from paddleocr import PaddleOCR
from fastapi.responses import JSONResponse
import os, re, cv2
from rapidfuzz import fuzz
from rapidfuzz import process
from typing import List
import pickle
import numpy as np
from google.cloud import translate_v2 as translate

app = FastAPI()

# Diccionario de ingredientes
ingredient_dict = ['Leche semidescremada', 'Sal', 'harina', 'huevo', 'leche', 'Cloruro de calcio', 'azucar', 'malta', 'maiz', 'pan']


# Cargar el archivo .pkl
nombre_archivo = "ingredients.pkl"
with open(nombre_archivo, 'rb') as archivo:
    datos = pickle.load(archivo)

# Convertir los datos a un array de numpy (si es necesario)
array_datos = np.array(datos)

@app.post("/translate/")
async def create_item(texto: str = Body(...)):
    result = await translate_text(ingredient_dict)
    return JSONResponse(content=result)


async def translate_text(texts: List[str]) -> List:
    """Translates text into the target language.
    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    results = []
    translate_client = translate.Client()

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.

    for item in texts:
        if isinstance(item, bytes):
            text = text.decode("utf-8")
        result = translate_client.translate(item, target_language="en", source_language="es")
        print("Text: {}".format(result["input"]))
        print("Translation: {}".format(result["translatedText"]))
        # print("Detected source language: {}".format(result["detectedSourceLanguage"]))
        results.append(result["translatedText"])

    return results

@app.post("/process_images/")
async def create_upload_files(files: list[UploadFile] = File(...)):

    img_ingredients = files[0]
    img_nutritional_facts = files[1]
    
    # Inicializar el modelo OCR
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

    sugar_percentage = await get_sugar_percentage(img_nutritional_facts, ocr_model)
    ingredients = await get_ingredients(img_ingredients, ocr_model)

    #traducción
    #ingredients_translated = translate(ingredients)

    #[0,0,0,1]

    #ingredientes_vector = getVector(ingredients_translated)

    reponse = {
        "ingredients": ingredients,
        "sugar": sugar_percentage
    }

    # TODO
    # result_nova = modelo.predict(ingredientes_vector)
    # result_impact_diabetes = model_desicion_tree.predict([result_nova, sugar_percentage])

    # response = {
    #     nova = result_nova,
    #     impact_diabetes = result_impact_diabetes
    # }

    result_example = {
        "nova": 3,
        "diabetes_risk": "Medio"
    }

    return JSONResponse(content=result_example)

async def get_sugar_percentage(img, ocr_model):
    img_path = f"./{img.filename}"
    # Guardar archivo
    with open(img_path, "wb") as f:
         f.write(await img.read())

    # Leer la imagen con OpenCV
    img_read = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)

    # Guardar la imagen en blanco y negro
    bw_img_path = f"./bw_{img.filename}"
    cv2.imwrite(bw_img_path, gray_img)

    result_nutritional_facts = ocr_model.ocr(bw_img_path)

    total_sugar = extract_total_sugar(result_nutritional_facts[0])
    text_sanitized = total_sugar.replace("g", "")
    # Convierte el resultado a un número flotante
    sugar_percentage = float(text_sanitized)
     # Eliminar los archivos después de procesarlos
    os.remove(img_path)
    os.remove(bw_img_path)
    print(sugar_percentage)
    return sugar_percentage


async def get_ingredients(img, ocr_model): 
    img_path = f"./{img.filename}"

    # Guardar archivo
    with open(img_path, "wb") as f:
         f.write(await img.read())

    # Leer la imagen con OpenCV
    img_read = cv2.imread(img_path)

    # Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)

    # Guardar la imagen en blanco y negro
    bw_img_path = f"./bw_{img.filename}"
    cv2.imwrite(bw_img_path, gray_img)
    # Obtener el texto de la imagen
    result = ocr_model.ocr(bw_img_path)
    os.remove(img_path)
    os.remove(bw_img_path)
    text_list = arrange_text(result[0])
    result_ingredients = search_ingredients(text_list)
    return result_ingredients


def search_ingredients(data):
    matches = set()
    for word in data:
        if word in ingredient_dict:
            matches.add(word)
        else:
            match = process.extractOne(word, array_datos, scorer=fuzz.ratio)
            if match and match[1] >= 80:
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

def extract_total_sugar(data):
    # Inicializar variables para almacenar el texto y las coordenadas del azúcar
    boxes_sugar = None
    text_sugar = None
    # Buscar el texto "azúcares totales" y obtener sus coordenadas
    for item in data:
        text = item[1][0].lower()
        if "azucares totales" in text:
            boxes_sugar = item[0]
            text_sugar = item[1][0]
            break
    # Si no se encuentra "azúcares totales", devolver None
    if not boxes_sugar:
        return None
    fe_sugar_y_coordinate = boxes_sugar[0]

    # Filtrar los elementos que están cerca de "azúcares totales" en el eje y
    values_sugar = [
        item for item in data
        if abs(item[0][0][1] - fe_sugar_y_coordinate) < 50 and item[1][0] != text_sugar
    ]
    # Encontrar el valor mínimo en [0][0][0] de los elementos filtrados
    corresponding_value = None
    min_valor = float('inf')
    
    for elemento in values_sugar:
        valor_actual = elemento[0][0][0]
        if valor_actual < min_valor:
            min_valor = valor_actual
            corresponding_value = elemento[1][0]  # Obtener el valor correspondiente

    return corresponding_value
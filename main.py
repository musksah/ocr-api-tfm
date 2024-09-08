from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import HTMLResponse
from paddleocr import PaddleOCR
from fastapi.responses import JSONResponse
import os, re, cv2
from typing import List
import numpy as np
from google.cloud import translate_v2 as translate
import spacy
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import json


# Descargar los recursos necesarios
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI()

# Ruta al archivo translations.json
file_path = 'translations.json'

# Cargar el archivo JSON
with open(file_path, 'r', encoding='utf-8') as file:
    translations = json.load(file)

async def translate_ingredients(ingredients):
    result = await translate_text(ingredients)
    return result

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
        if item not in translations:    
            result = translate_client.translate(item, target_language="en", source_language="es")
            translations[item] = result["translatedText"]
            print(f"Agregado: {item} -> {result['translatedText']}")
            results.append((item, result["translatedText"]))
        else:
            existing_value = translations[item]
            results.append((item,existing_value))
            print(f"'{item}' ya existe, no se agregará.")
    
    # Guardar el archivo JSON actualizado
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(translations, file, ensure_ascii=False, indent=4)

    return results

@app.post("/process_images/")
async def create_upload_files(files: list[UploadFile] = File(...)):

    img_ingredients = files[0]
    img_nutritional_facts = files[1]
    
    # Inicializar el modelo OCR
    ocr_model = PaddleOCR(use_angle_cls=True, lang='es')

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
        "diabetes_impact": "Medio"
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

    text_sugar = extract_total_sugar(result_nutritional_facts[0])
    total_sugar = get_sugar_value(text_sugar)
    
     # Eliminar los archivos después de procesarlos
    os.remove(img_path)
    os.remove(bw_img_path)
    return total_sugar


async def get_ingredients(img, ocr_model): 
    img_path = f"./{img.filename}"

    # Guardar archivo
    with open(img_path, "wb") as f:
         f.write(await img.read())

    # Leer la imagen con OpenCV
    img_read = cv2.imread(img_path)

    # Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)

    # Dilatar para engrosar las letras
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # O usar erosión si las letras están demasiado unidas
    imagen_erosionada = cv2.erode(gray_img, kernel, iterations=1)

    # Guardar la imagen en blanco y negro
    bw_img_path = f"./bw_{img.filename}"
    cv2.imwrite(bw_img_path, imagen_erosionada)
    # Obtener el texto de la imagen
    result = ocr_model.ocr(bw_img_path)
    print("Result ingredients: ")
    print(result)
    os.remove(img_path)
    os.remove(bw_img_path)
    text_list = preprocess_text(result[0])
    final_texts = await translate_ingredients(text_list)
    print("final_texts: ")
    print(final_texts)
    return final_texts


def get_sugar_value(text_sugar):
    try:
        if(text_sugar):
            text_sugar.replace("g", "")
            total_sugar = float(text_sugar)
            return total_sugar
        return None
    except Exception as e:
        if(text_sugar):
            return 0
        return None


def preprocess_text(data):
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

    texts_sanitized = [
        texto.strip() for texto in result_texts 
        if not re.search(r'\d', texto)
        and texto.strip() 
        and len(texto) > 1
        and "ingrediente" not in texto.lower()
    ]

    textos = filtrar_textos_nltk(texts_sanitized)
    return textos

def filtrar_textos(textos):
    nlp = spacy.load('es_core_news_sm')
    textos_filtrados = []

    for texto in textos:
        # Procesar el texto
        doc = nlp(texto)

        # Imprimir los tipos de tokens y sus etiquetas
        for token in doc:
            print(f"Texto original: {texto}, Token: {token.text}, Tipo: {token.pos_}, Etiqueta detallada: {token.tag_}")

        # Verificar si el texto contiene verbos
        has_verbs = any(token.pos_ == 'VERB' for token in doc)
        
        if has_verbs:
            continue

        # no contienen números ni son de una sola letra, y convertir los sustantivos a singular
        if any(token.pos_ in {'NOUN', 'PROPN'} for token in doc):
            textos_filtrados.append(texto)

    return textos_filtrados

def filtrar_textos_nltk(textos):
    textos_filtrados = []
    
    lemmatizer = nltk.WordNetLemmatizer()
    
    for texto in textos:
        # Tokenizar el texto
        tokens = word_tokenize(texto)
        
        # Etiquetado POS de NLTK
        tagged_tokens = pos_tag(tokens)
        
        # Filtrar por verbos
        has_verbs = any(tag.startswith('V') for word, tag in tagged_tokens)
        
        if has_verbs:
            continue

        # Filtrar por sustantivos y nombres propios
        contains_noun_or_propn = any(tag.startswith('N') for word, tag in tagged_tokens)
        
        if contains_noun_or_propn:
            # Lematizar sustantivos
            lemmatized_text = []
            for word, tag in tagged_tokens:
                wordnet_pos = get_wordnet_pos(tag)
                if wordnet_pos == wordnet.NOUN:
                    lemmatized_text.append(lemmatizer.lemmatize(word, pos=wordnet.NOUN))
                else:
                    lemmatized_text.append(word)
            
            # Unir los tokens lematizados y agregarlos a los textos filtrados
            textos_filtrados.append(" ".join(lemmatized_text))
        
        # Imprimir los tipos de tokens y sus etiquetas
        for word, tag in tagged_tokens:
            print(f"Texto original: {texto}, Token: {word}, Tipo: {tag}")
    
    return textos_filtrados

# Conversión de etiquetas de NLTK a etiquetas de WordNet para lematización
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

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
    fe_sugar_x_coordinate,fe_sugar_y_coordinate = boxes_sugar[0]

    # Filtrar los elementos que están cerca de "azúcares totales" en el eje y
    values_sugar = [
        item for item in data if abs(item[0][0][1] - fe_sugar_y_coordinate) < 50 and item[1][0] != text_sugar
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
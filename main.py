from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from paddleocr import PaddleOCR
from fastapi.responses import JSONResponse
import os, re, cv2

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):

    img_path = f"./{file.filename}"

    # Guardar archivo
    with open(img_path, "wb") as f:
         f.write(await file.read())

    # Leer la imagen con OpenCV
    img = cv2.imread(img_path)

    # Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Guardar la imagen en blanco y negro
    bw_img_path = f"./bw_{file.filename}"
    cv2.imwrite(bw_img_path, gray_img)

    # Inicializar el modelo OCR
    ocr_model = PaddleOCR(lang='en')

    # Obtener el texto de la imagen
    result = ocr_model.ocr(bw_img_path)

    # El output de paddleocr que has proporcionado
    ocr_output = result[0]

    # Eliminar los archivos después de procesarlos
    os.remove(img_path)
    os.remove(bw_img_path)

    # Lista para almacenar los ingredientes
    ingredientes = []

    # Bandera para indicar si estamos en la sección de ingredientes
    en_ingredientes = False

    # Palabras clave para identificar la sección de ingredientes
    keywords = ["INGREDIENTES"]

    # Recorre las detecciones del OCR
    for item in ocr_output:
        texto = item[1][0]
        if any(keyword in texto for keyword in keywords):
            en_ingredientes = True
        if en_ingredientes:
            ingredientes.append(texto)
        # Verificar si hemos llegado al final de la sección de ingredientes
        if en_ingredientes and texto.endswith("."):
            en_ingredientes = False
            break

    # Unir los ingredientes en una sola cadena y luego dividirlos por comas
    ingredientes_completos = " ".join(ingredientes)
    lista_ingredientes = re.split(r',\s*', ingredientes_completos)

    return JSONResponse(content=lista_ingredientes)


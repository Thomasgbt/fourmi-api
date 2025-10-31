import os
import torch
import json
import csv
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from PIL import Image
from torchvision import models, transforms
from shapely.geometry import shape, MultiPolygon, Point
import io
import time

# =======================================================
# ✅ CONFIGURATION FASTAPI
# =======================================================
app = FastAPI(title="Fourmi API - Prédiction")

# Autoriser tout pour test (à restreindre ensuite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# ✅ VARIABLES GLOBALES
# =======================================================
cities_dict, geo_dict, size_dict, date_dict = {}, {}, {}, {}
model_ailes = model_sexe = model_caste = model_role = None
model_espece1 = model_espece2 = model_espece3 = None
classes_espece1 = classes_espece2 = classes_espece3 = []
img_size = 320

# =======================================================
# 🔹 FONCTIONS DE BASE (simplifiées)
# =======================================================
def safe_load_model(path, with_classes=True):
    """Charge un modèle si le fichier existe, sinon None"""
    if not os.path.exists(path):
        print(f"⚠️ Modèle manquant : {path}")
        return (None, [])
    try:
        device = torch.device("cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if with_classes:
            classes = checkpoint.get("class_names", [])
            num_classes = len(classes) if classes else 2
        else:
            classes = []
            num_classes = 2
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.eval()
        print(f"✅ Modèle chargé : {os.path.basename(path)}")
        return (model, classes)
    except Exception as e:
        print(f"❌ Erreur chargement modèle {path}: {e}")
        return (None, [])

def basic_transform(img):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img)

# =======================================================
# 🔹 STARTUP (chargement données et modèles)
# =======================================================
@app.on_event("startup")
def startup_event():
    global model_ailes, model_sexe, model_caste, model_role
    global model_espece1, model_espece2, model_espece3
    global classes_espece1, classes_espece2, classes_espece3

    print("🚀 Initialisation des modèles et données...")

    model_ailes, _ = safe_load_model("Modeles/Model_Ailes_27_08.pth", with_classes=False)
    model_sexe, _ = safe_load_model("Modeles/Model_MaleGyne_28_08.pth", with_classes=False)
    model_caste, _ = safe_load_model("Modeles/Model_GyneNoGyne_28_08.pth", with_classes=False)
    model_role, _ = safe_load_model("Modeles/Model_CastOuvriere_06_09.pth", with_classes=False)
    model_espece1, classes_espece1 = safe_load_model("Modeles/Model_SpeciesGyne_28_08.pth")
    model_espece2, classes_espece2 = safe_load_model("Modeles/Model_SpeciesOuvriere_24_08.pth")
    model_espece3, classes_espece3 = safe_load_model("Modeles/Model_SpeciesSoldat_17_09.pth")

    print("✅ Initialisation terminée.")

# =======================================================
# 🔹 ENDPOINT RACINE
# =======================================================
@app.get("/")
def read_root():
    return {"message": "🚀 API Fourmi opérationnelle sur Render !"}

# =======================================================
# 🔹 ENDPOINT : LISTE DES ESPECES
# =======================================================
@app.get("/species")
def get_species():
    all_species = set(classes_espece1 + classes_espece2 + classes_espece3)
    return {"species": sorted(list(all_species))}

# =======================================================
# 🔹 ENDPOINT : PREDICTION (simplifiée)
# =======================================================
@app.post("/predict")
async def predict(
    images: List[UploadFile] = File(...),
    ailes: Optional[str] = Form(None),
    genre: Optional[str] = Form(None),
    caste: Optional[str] = Form(None)
):
    print(f"📸 Nombre d’images reçues : {len(images)}")
    t0 = time.time()

    results = []
    for idx, img_file in enumerate(images):
        img_bytes = await img_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = basic_transform(img).unsqueeze(0)
        # 🔹 Si un modèle existe, on fait une prédiction factice (démo)
        if model_ailes:
            with torch.no_grad():
                output = model_ailes(tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, cls = torch.max(probs, 1)
                result = f"Image {idx+1}: {['Oui', 'Non'][cls.item()]} ({conf.item()*100:.1f}%)"
        else:
            result = f"Image {idx+1}: Modèle indisponible"
        results.append(result)

    return JSONResponse(content={
        "results": results,
        "ailes": ailes,
        "genre": genre,
        "caste": caste,
        "duration": round(time.time() - t0, 2)
    })

# =======================================================
# 🔹 MAIN (pour Render)
# =======================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)


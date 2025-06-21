import os
import pickle
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from typing import Dict
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
app = FastAPI()

# --- Existing Code Begin ---
PROPERTY_FILE = "output (2).txt"
property_dicts = None

class PredictionRequest(BaseModel):
    sequence: str
    organism: str
def load_models_from_huggingface():
    repo_id = "rajesh500759/allorganism"
    subdir = "allorganism/best_property"
    model_filenames = [
        "AdaBoost.pkl",
        "Bagging.pkl",
        "CatBoost.cbm",
        "Decision_Tree.pkl",
        "Extra_Trees.pkl",
        "Gradient_Boosting.pkl",
        "K-NN.pkl",
        "LightGBM.pkl",
        "Logistic_Regression.pkl",
        "Naive_Bayes.pkl",
        "Perceptron.pkl",
        "Random_Forest.pkl",
        "SGD.pkl",
        "SVM.pkl",
        "XGBoost.json"
    ]

    models = {}
    for filename in model_filenames:
        model_name, ext = os.path.splitext(filename)
        try:
            # This downloads the file to local cache and returns path
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{subdir}/{filename}",
                cache_dir="huggingface_cache"  # Optional: customize cache directory
            )

            if ext == ".pkl":
                with open(model_path, "rb") as f:
                    models[model_name] = pickle.load(f)

            elif ext == ".json":
                model = XGBClassifier()
                model.load_model(model_path)
                models[model_name] = model

            elif ext == ".cbm":
                model = CatBoostClassifier()
                model.load_model(model_path)
                models[model_name] = model

            print(f"Loaded model: {model_name}")

        except Exception as e:
            print(f"Error loading {filename} from Hugging Face: {e}")

    if not models:
        raise RuntimeError("No models could be loaded from Hugging Face")

    return models
def load_properties():
    """Load and process DNA property data with numeric conversion"""
    try:
        with open(PROPERTY_FILE) as f:
            text = f.read().replace('\t', ' ')
        lines = [line.split() for line in text.split('\n')[1:126] if line.strip()]
        columns = text.split('\n')[0].split()
        df = pd.DataFrame(lines, columns=columns)
        df.set_index(df.columns[0], inplace=True) 
        df = df.apply(pd.to_numeric, errors='coerce')
        df.fillna(0.0, inplace=True)
        return {str(prop): df.loc[prop].to_dict() for prop in df.index}
    except Exception as e:
        raise RuntimeError(f"Failed to load properties: {str(e)}")

try:
    property_dicts = load_properties()
    print(f"Successfully loaded {len(property_dicts)} properties")
except Exception as e:
    print(f"Fatal error during property initialization: {str(e)}")
    raise SystemExit(1)

def encode_sequence(sequence, prop_name):
    """Encode DNA sequence using 2-mer sliding window with numeric validation"""
    if not property_dicts:
        raise ValueError("Property database not initialized")
    prop_name = str(prop_name)
    if prop_name not in property_dicts:
        raise ValueError(f"Property '{prop_name}' not found in database")
    encoded = []
    prop_dict = property_dicts[prop_name]
    for i in range(len(sequence)-1):
        dimer = sequence[i:i+2].upper()
        value = prop_dict.get(dimer, 0.0)
        encoded.append(float(value))
    return encoded

def preprocess_sequence(sequence, prop_name):
    """Preprocess input sequence into numerical representation"""
    if len(sequence) != 150:
        raise ValueError("Sequence must be exactly 150 bases long")
    sequence = sequence.upper()
    if any(base not in "ATGC" for base in sequence):
        raise ValueError("Invalid bases in sequence. Only A, T, G, C allowed")
    encoded = encode_sequence(sequence, prop_name)
    if len(encoded) != 149:
        raise ValueError(f"Encoding failed: Expected 149 features, got {len(encoded)}")
    return np.array(encoded, dtype=np.float32).reshape(1, -1)

def load_models(organism):
    """Load all trained models for the given organism"""
    model_dir = f"models/{organism}/best_property"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"No models found for {organism}")
    models = {}
    for filename in os.listdir(model_dir):
        model_path = os.path.join(model_dir, filename)
        model_name, ext = os.path.splitext(filename)
        try:
            if ext == ".pkl":
                with open(model_path, "rb") as f:
                    models[model_name] = pickle.load(f)
            elif ext == ".json":
                model = XGBClassifier()
                model.load_model(model_path)
                models[model_name] = model
            elif ext == ".cbm":
                model = CatBoostClassifier()
                model.load_model(model_path)
                models[model_name] = model
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
    if not models:
        raise RuntimeError("All models failed to load")
    return models

@app.post("/predict")
def predict(request: PredictionRequest):
    """API endpoint to predict promoter sequences using multiple models"""
    try:
        print("c")
        organism_dir = f"models/{request.organism}"
        print("c1")
        prop_rankings = pd.read_csv(os.path.join(organism_dir, f"{request.organism}_property_rankings.csv"))
        print("c2")
        prop_rankings['property'] = prop_rankings['property'].astype(str)
        print("cc")
        best_property = prop_rankings.iloc[0]['property']
        print("c")
        processed_seq = preprocess_sequence(request.sequence, best_property)
        models = load_models_from_huggingface()
        predictions: Dict[str, int] = {}
        for model_name, model in models.items():
            pred = model.predict(processed_seq)
            predictions[model_name] = int(pred[0])
        votes = sum(predictions.values())
        final_prediction = 1 if votes > len(predictions) / 2 else 0
        report = [
            f"Sequence: {request.sequence}",
            f"Organism: {request.organism}",
            f"Best Property: {best_property}",
            "\nModel Predictions:"
        ]
        for model, pred in predictions.items():
            report.append(f"{model}: {'Promoter' if pred == 1 else 'Non-promoter'}")
        report.append(f"\nFinal Ensemble Prediction: {'Promoter' if final_prediction == 1 else 'Non-promoter'} ({votes}/{len(predictions)} votes)")
        os.makedirs("reports", exist_ok=True)
        report_path = os.path.join("reports", f"{request.organism}_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        return {
            "organism": request.organism,
            "best_property": best_property,
            "predictions": predictions,
            "ensemble_prediction": final_prediction,
            "report_path": report_path
        }
    except ValueError as ve:
        print(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        print(f"File error: {str(fe)}")
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        print(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
# --- Existing Code End ---


# --- New Function Implementation Begin ---
@app.post("/predict_regions")
def predict_regions(request: PredictionRequest):
    try:
        sequence = request.sequence.strip().upper()
        if len(sequence) < 150:
            raise HTTPException(status_code=400, detail="Input sequence must be at least 150 bases long.")

        organism = request.organism

        # Load precision/recall metrics
        metrics_file = os.path.join("models", organism, "best_property", f"{organism}_best_property_metrics.csv")
        print("hello")
        if not os.path.exists(metrics_file):
            raise HTTPException(status_code=404, detail=f"Metrics file not found: {metrics_file}")
        
        df_metrics = pd.read_csv(metrics_file)
        best_precision = df_metrics["precision"].max()
        best_recall = df_metrics["recall"].max()

        seq_length = len(sequence)
        score_profile = [0.0] * seq_length

        # Slide 150-base window and directly call `predict` function
        num_windows = seq_length - 150 + 1
        for i in range(num_windows):
            window_seq = sequence[i:i+150]
            predict_request = PredictionRequest(sequence=window_seq, organism=organism)
            try:
                result = predict(predict_request)  # 🔥 Direct internal function call
                is_promoter = result.get("ensemble_prediction", 0) == 1
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error obtaining prediction for window at index {i}: {str(e)}")

            for j in range(i, i + 150):
                score_profile[j] += best_precision if is_promoter else -best_recall

        # Extract positive-score promoter regions of length > 10
        promoter_regions = []
        in_region = False
        region_start = 0
        for idx, score in enumerate(score_profile):
            if score > 0:
                if not in_region:
                    in_region = True
                    region_start = idx
            else:
                if in_region:
                    region_end = idx - 1
                    if region_end - region_start + 1 >= 10:
                        promoter_regions.append({
                            "start": region_start,
                            "end": region_end,
                            "region_sequence": sequence[region_start:region_end+1]
                        })
                    in_region = False

        if in_region:
            region_end = seq_length - 1
            if region_end - region_start + 1 >= 10:
                promoter_regions.append({
                    "start": region_start,
                    "end": region_end,
                    "region_sequence": sequence[region_start:region_end+1]
                })

        return {
            "organism": organism,
            "input_sequence_length": seq_length,
            "best_precision": best_precision,
            "best_recall": best_recall,
            "score_profile": score_profile,
            "promoter_regions": promoter_regions
        }

    except Exception as e:
        print("The error in the code is:", e)
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT env var
    uvicorn.run(app, host="0.0.0.0", port=port)
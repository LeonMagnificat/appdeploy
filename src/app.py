import os
import shutil
import zipfile
import io
import json
import warnings
import logging
from datetime import datetime, timedelta
from typing import List

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration for Railway
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")  # Fallback to SQLite if not set
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+mysqlconnector://", 1)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')
# Suppress HDF5 legacy warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*HDF5 file via `model.save()`.*")

# Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    predictions = relationship("Prediction", back_populates="user")
    retrainings = relationship("Retraining", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    predicted_disease = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="predictions")

class Retraining(Base):
    __tablename__ = "retrainings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    num_classes = Column(Integer, nullable=False)
    training_accuracy = Column(Float)
    validation_accuracy = Column(Float, nullable=True)
    class_metrics = Column(Text)  # Store JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="retrainings")

Base.metadata.create_all(bind=engine)

# Define base directory and paths for Railway (assuming code is in src/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_DIR = os.path.join(BASE_DIR, "data")
VISUALIZATION_DIR = os.path.join(BASE_DIR, "static/visualizations")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.h5")

# Create directories upfront
for directory in [UPLOAD_DIR, VISUALIZATION_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Mount static files for visualizations
app.mount("/static/visualizations", StaticFiles(directory=VISUALIZATION_DIR), name="visualizations")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Does the model file exist? {os.path.exists(MODEL_PATH)}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Define initial class names for plant diseases
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Pydantic models for request/response
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(db, username=username)
    if user is None:
        raise credentials_exception
    
    return user

def preprocess_image(img_bytes: bytes):
    """Preprocess image for prediction."""
    img = image.load_img(io.BytesIO(img_bytes), target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_zip(zip_path, extract_to):
    """Extract ZIP files."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def save_visualizations(y_true, y_pred_classes, target_names, history=None):
    """Save enhanced visualizations including classification report, confusion matrix, and training plots."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # 1. Classification Report
    class_report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True)
    
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for cls in target_names:
        if cls in class_report:
            rows.append([
                cls,
                f"{class_report[cls]['precision']:.2f}",
                f"{class_report[cls]['recall']:.2f}",
                f"{class_report[cls]['f1-score']:.2f}",
                f"{class_report[cls]['support']}"
            ])
    total_support = sum(class_report[cls]['support'] for cls in target_names if cls in class_report)
    rows.append(["Accuracy", "", "", f"{class_report['accuracy']:.2f}", f"{total_support}"])

    fig, ax = plt.subplots(figsize=(12, len(target_names) * 0.6 + 2))
    ax.axis('off')
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#4CAF50'] * len(headers),
        colWidths=[0.4, 0.15, 0.15, 0.15, 0.15],
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4CAF50')
        else:
            cell.set_text_props(color='black')
            cell.set_facecolor('#F5F5F5' if row % 2 == 0 else '#FFFFFF')
        cell.set_edgecolor('#D3D3D3')
    
    plt.title("Classification Report", fontsize=18, weight='bold', pad=20, color='#333333')
    plt.savefig(
        os.path.join(VISUALIZATION_DIR, f"classification_report_{timestamp}.png"),
        bbox_inches='tight',
        dpi=300,
        facecolor='white'
    )
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(max(10, len(target_names)), max(10, len(target_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, cbar=True)
    plt.title("Confusion Matrix", fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"confusion_matrix_{timestamp}.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # 3. Training and Validation Loss
    if history and 'loss' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, f"loss_plot_{timestamp}.png"), bbox_inches='tight', dpi=300)
        plt.close()

    # 4. Training and Validation Accuracy
    if history and 'accuracy' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, f"accuracy_plot_{timestamp}.png"), bbox_inches='tight', dpi=300)
        plt.close()
    
    return timestamp

# Authentication endpoints
@app.post("/signup", response_model=Token)
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, username=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Protected endpoints
@app.post("/predict")
async def predict(file: UploadFile = File(...),
                 db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    img_bytes = await file.read()
    img = preprocess_image(img_bytes)
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    disease = CLASS_NAMES[predicted_index]
    
    prediction = Prediction(
        user_id=current_user.id,
        predicted_disease=disease,
        confidence=float(confidence)
    )
    db.add(prediction)
    db.commit()
    
    return JSONResponse(content={"prediction": disease, "confidence": float(confidence)})

@app.post("/retrain")
async def retrain(files: List[UploadFile] = File(...),
                 learning_rate: float = 0.0001,
                 epochs: int = 10,
                 db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    global model, CLASS_NAMES
    
    new_data_dir = os.path.join(UPLOAD_DIR, "new_data")
    os.makedirs(new_data_dir, exist_ok=True)
    
    extracted_dirs = []
    
    try:
        # 1. Process uploaded files
        for file in files:
            if not file.filename.lower().endswith('.zip'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} must be a ZIP file")
                
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            logger.info(f"Saving uploaded file to {file_path}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            extract_dir = os.path.join(UPLOAD_DIR, f"extract_{os.path.splitext(file.filename)[0]}")
            os.makedirs(extract_dir, exist_ok=True)
            logger.info(f"Extracting {file_path} to {extract_dir}")
            try:
                extract_zip(file_path, extract_dir)
                extracted_dirs.append(extract_dir)
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {file.filename}")
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed temporary file {file_path}")
            
            # Check if expected subdirs exist
            found_subdirs = False
            for subdir in ['train', 'val', 'test']:
                subdir_path = os.path.join(extract_dir, subdir)
                if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                    found_subdirs = True
                    logger.info(f"Processing {subdir} in {extract_dir}")
                    for class_name in os.listdir(subdir_path):
                        class_path = os.path.join(subdir_path, class_name)
                        if os.path.isdir(class_path) and class_name not in ['__MACOSX']:
                            target_dir = os.path.join(new_data_dir, class_name)
                            os.makedirs(target_dir, exist_ok=True)
                            for img in os.listdir(class_path):
                                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    src = os.path.join(class_path, img)
                                    dst = os.path.join(target_dir, img)
                                    shutil.copy(src, dst)
                                    logger.debug(f"Copied {src} to {dst}")
            
            if not found_subdirs:
                raise HTTPException(status_code=400, detail=f"ZIP file {file.filename} must contain 'train', 'val', or 'test' subdirectories")

        # 2. Filter classes with sufficient data
        class_counts = {}
        for class_dir in os.listdir(new_data_dir):
            class_path = os.path.join(new_data_dir, class_dir)
            if os.path.isdir(class_path):
                image_count = len([f for f in os.listdir(class_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if image_count >= 2:
                    class_counts[class_dir] = image_count
                else:
                    shutil.rmtree(class_path)
                    logger.info(f"Removed class {class_dir} with insufficient samples ({image_count})")
        
        if not class_counts:
            raise HTTPException(status_code=400, detail={
                "error": "No valid classes with sufficient data found",
                "details": "Each class must have at least 2 images"
            })
        
        logger.info(f"Valid classes found: {class_counts}")
        
        # 3. Create data generators
        target_names = list(class_counts.keys())
        all_classes = list(set(CLASS_NAMES + target_names))
        use_validation = all(count >= 4 for count in class_counts.values())
        
        if use_validation:
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )
            train_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                classes=all_classes,
                subset='training',
                shuffle=True
            )
            validation_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                classes=all_classes,
                subset='validation',
                shuffle=False
            )
        else:
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )
            train_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                classes=all_classes,
                shuffle=True
            )
            validation_generator = None
        
        # 4. Create a new model for fine-tuning
        temp_model_path = os.path.join(MODEL_DIR, "temp_model.h5")
        logger.info(f"Saving temporary model to {temp_model_path}")
        model.save(temp_model_path)
        working_model = tf.keras.models.load_model(temp_model_path)
        
        num_layers = len(working_model.layers)
        freeze_until = int(num_layers * 0.98)
        for i, layer in enumerate(working_model.layers):
            layer.trainable = i >= freeze_until
        
        working_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 5. Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if use_validation else 'loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if use_validation else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        # 6. Train the model
        logger.info("Starting model training")
        if use_validation:
            history = working_model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=max(1, len(train_generator)),
                validation_steps=max(1, len(validation_generator))
            )
        else:
            history = working_model.fit(
                train_generator,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=max(1, len(train_generator))
            )
        
        # 7. Generate classification report and predictions
        if use_validation:
            validation_generator.reset()
            y_pred = working_model.predict(validation_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = validation_generator.classes
        else:
            train_generator.reset()
            y_pred = working_model.predict(train_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = train_generator.classes
        
        class_report = classification_report(
            y_true,
            y_pred_classes,
            target_names=target_names,
            output_dict=True
        )
        
        # 8. Save visualizations with history
        timestamp = save_visualizations(y_true, y_pred_classes, target_names, history)
        
        # 9. Save the fine-tuned model in .h5 format
        logger.info(f"Saving fine-tuned model to {MODEL_PATH}")
        working_model.save(MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)
        
        CLASS_NAMES = all_classes
        with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
            json.dump(CLASS_NAMES, f)
        
        # 10. Prepare metrics and save to database
        class_metrics = {}
        for class_name in target_names:
            if class_name in class_report:
                class_metrics[class_name] = {
                    "precision": float(class_report[class_name]['precision']),
                    "recall": float(class_report[class_name]['recall']),
                    "f1_score": float(class_report[class_name]['f1-score']),
                    "support": int(class_report[class_name]['support'])
                }
        
        new_classes_added = [cls for cls in target_names if cls not in CLASS_NAMES]
        
        training_accuracy = float(history.history['accuracy'][-1]) if 'accuracy' in history.history else None
        validation_accuracy = float(history.history['val_accuracy'][-1]) if use_validation and 'val_accuracy' in history.history else None
        
        retraining = Retraining(
            user_id=current_user.id,
            num_classes=len(CLASS_NAMES),
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            class_metrics=json.dumps(class_metrics)
        )
        db.add(retraining)
        db.commit()
        
        # 11. Prepare response with Railway-compatible URLs
        base_url = os.getenv("RAILWAY_PUBLIC_DOMAIN", "http://localhost:8000")
        response_content = {
            "message": "Model fine-tuning successful!",
            "num_classes": len(CLASS_NAMES),
            "new_classes_added": new_classes_added,
            "class_counts": class_counts,
            "training_accuracy": training_accuracy,
            "class_metrics": class_metrics,
            "fine_tuned_model_path": MODEL_PATH,
            "visualization_files": {
                "classification_report": f"{base_url}/static/visualizations/classification_report_{timestamp}.png",
                "confusion_matrix": f"{base_url}/static/visualizations/confusion_matrix_{timestamp}.png",
                "loss_plot": f"{base_url}/static/visualizations/loss_plot_{timestamp}.png",
                "accuracy_plot": f"{base_url}/static/visualizations/accuracy_plot_{timestamp}.png"
            },
            "retraining_id": retraining.id,
            "user_id": current_user.id
        }
        
        if use_validation:
            response_content["validation_accuracy"] = validation_accuracy
        
        logger.info("Retraining completed successfully")
        return JSONResponse(content=response_content)
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        import traceback
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(content={
            "error": str(e),
            "details": traceback.format_exc()
        }, status_code=500)
    
    finally:
        # Robust cleanup
        for extract_dir in extracted_dirs:
            if os.path.exists(extract_dir):
                try:
                    shutil.rmtree(extract_dir)
                    logger.info(f"Cleaned up {extract_dir}")
                except Exception as e:
                    logger.error(f"Failed to remove {extract_dir}: {e}")
        if os.path.exists(new_data_dir):
            try:
                shutil.rmtree(new_data_dir)
                logger.info(f"Cleaned up {new_data_dir}")
            except Exception as e:
                logger.error(f"Failed to remove {new_data_dir}: {e}")
        temp_model_path = os.path.join(MODEL_DIR, "temp_model.h5")
        if os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
                logger.info(f"Cleaned up {temp_model_path}")
            except Exception as e:
                logger.error(f"Failed to remove {temp_model_path}: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Plant Disease Prediction API!"}

@app.get("/prediction_history")
async def get_prediction_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    predictions = db.query(Prediction).filter(Prediction.user_id == current_user.id).order_by(Prediction.timestamp.desc()).all()
    return [{"id": p.id, "text": f"Predicted disease: {p.predicted_disease}", "confidence": p.confidence, "date": p.timestamp.isoformat()}
            for p in predictions]

@app.get("/retraining_history")
async def get_retraining_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    retrainings = db.query(Retraining).filter(Retraining.user_id == current_user.id).order_by(Retraining.timestamp.desc()).all()
    return [{"id": r.id, "text": f"Retrained model with {r.num_classes} classes", "training_accuracy": r.training_accuracy,
             "validation_accuracy": r.validation_accuracy, "class_metrics": json.loads(r.class_metrics),
             "date": r.timestamp.isoformat()}
            for r in retrainings]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Railway's PORT env var
    uvicorn.run(app, host="0.0.0.0", port=port)
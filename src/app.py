import os
import shutil
import zipfile
import io
import json
import warnings
from datetime import datetime
from typing import List, Any

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

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
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

# Initialize FastAPI app
app = FastAPI()

# Serve static files from the "visualizations" directory
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "plant_disease_model.keras"
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

UPLOAD_DIR = "Data"
VISUALIZATION_DIR = "visualizations"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

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
    from datetime import timedelta
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

def extract_zip(zip_path, extract_to):
    """Extract ZIP files."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def save_visualizations(y_true, y_pred_classes, target_names):
    """Save classification report and confusion matrix as PNG files."""
    class_report = classification_report(y_true, y_pred_classes, target_names=target_names)
    
    plt.figure(figsize=(10, len(target_names) * 0.5 + 2))
    plt.text(0.01, 0.99, class_report, {'fontsize': 10}, fontfamily='monospace')
    plt.axis('off')
    plt.title("Classification Report")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "classification_report.png"),
                bbox_inches='tight', dpi=300)
    plt.close()

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(max(10, len(target_names)), max(10, len(target_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "confusion_matrix.png"),
                bbox_inches='tight', dpi=300)
    plt.close()

@app.post("/retrain")
async def retrain(files: List[UploadFile] = File(...),
                 learning_rate: float = 0.0001,
                 epochs: int = 10,
                 db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    global model, CLASS_NAMES
    
    new_data_dir = os.path.join(UPLOAD_DIR, "new_data")
    os.makedirs(new_data_dir, exist_ok=True)
    
    try:
        # 1. Process uploaded files
        extracted_dirs = []
        
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            if file.filename.endswith(".zip"):
                extract_dir = os.path.join(UPLOAD_DIR, f"extract_{os.path.splitext(file.filename)[0]}")
                os.makedirs(extract_dir, exist_ok=True)
                extract_zip(file_path, extract_dir)
                extracted_dirs.append(extract_dir)
                os.remove(file_path)
                
                print(f"Extracted ZIP to: {extract_dir}")
                print(f"Contents of extract_dir: {os.listdir(extract_dir)}")
                
                # Process train and val subdirectories
                for subdir in ['train', 'val','test']:
                    subdir_path = os.path.join(extract_dir, subdir)
                    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                        print(f"Processing {subdir}: {os.listdir(subdir_path)}")
                        for class_name in os.listdir(subdir_path):
                            class_path = os.path.join(subdir_path, class_name)
                            if os.path.isdir(class_path) and class_name not in ['__MACOSX']:
                                target_dir = os.path.join(new_data_dir, class_name)
                                os.makedirs(target_dir, exist_ok=True)
                                for img in os.listdir(class_path):
                                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                                        shutil.copy(os.path.join(class_path, img), os.path.join(target_dir, img))
        
        # 2. Filter classes with sufficient data
        class_counts = {}
        for class_dir in os.listdir(new_data_dir):
            class_path = os.path.join(new_data_dir, class_dir)
            if os.path.isdir(class_path):
                image_count = len([f for f in os.listdir(class_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if image_count >= 2:  # Minimum 2 images per class
                    class_counts[class_dir] = image_count
                else:
                    print(f"Skipping class {class_dir} with insufficient samples ({image_count})")
                    shutil.rmtree(class_path)
        
        if not class_counts:
            raise HTTPException(status_code=400, detail={
                "error": "No valid classes with sufficient data found",
                "details": "Each class must have at least 2 images",
                "class_counts": {class_dir: len(os.listdir(os.path.join(new_data_dir, class_dir)))
                               for class_dir in os.listdir(new_data_dir) if os.path.isdir(os.path.join(new_data_dir, class_dir))}
            })
        
        print(f"Valid classes and image counts: {class_counts}")
        
        # 3. Create data generators
        target_names = list(class_counts.keys())
        all_classes = list(set(CLASS_NAMES + target_names))
        use_validation = all(count >= 4 for count in class_counts.values())  # Require 4+ for validation split
        
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
        temp_model_path = os.path.join(os.path.dirname(MODEL_PATH), "temp_model.keras")
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
        
        # 8. Save visualizations
        save_visualizations(y_true, y_pred_classes, target_names)
        
        # 9. Save the fine-tuned model
        fine_tuned_model_path = os.path.join(os.path.dirname(MODEL_PATH), "plant_disease_model.keras")
        working_model.save(fine_tuned_model_path)
        model = tf.keras.models.load_model(fine_tuned_model_path)
        
        CLASS_NAMES = all_classes
        with open(os.path.join(os.path.dirname(MODEL_PATH), "class_names.json"), "w") as f:
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
        
        # 11. Prepare response
        base_url = "http://127.0.0.1:8000"  # Adjust for production
        response_content = {
            "message": "Model fine-tuning successful!",
            "num_classes": len(CLASS_NAMES),
            "new_classes_added": new_classes_added,
            "class_counts": class_counts,
            "training_accuracy": training_accuracy,
            "class_metrics": class_metrics,
            "fine_tuned_model_path": fine_tuned_model_path,
            "visualization_files": {
                "classification_report": f"{base_url}/visualizations/classification_report.png",
                "confusion_matrix": f"{base_url}/visualizations/confusion_matrix.png"
            },
            "retraining_id": retraining.id,
            "user_id": current_user.id
        }
        
        if use_validation:
            response_content["validation_accuracy"] = validation_accuracy
        
        return JSONResponse(content=response_content)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        return JSONResponse(content={
            "error": str(e),
            "details": traceback.format_exc()
        }, status_code=500)
    
    finally:
        for extract_dir in extracted_dirs:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
        if os.path.exists(new_data_dir):
            shutil.rmtree(new_data_dir)
        temp_model_path = os.path.join(os.path.dirname(MODEL_PATH), "temp_model.keras")
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

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
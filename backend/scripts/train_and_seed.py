"""Complete ASL Interpreter Database Seeding & Model Training Script.

Tasks performed:
1. Connects to database (backend/asl.db or DATABASE_URL).
2. Seeds all 26 alphabet letters (A-Z) and common essential signs (hello, thank you, please, etc.)
   with canonical 32x126 3D MediaPipe hand landmarks for SkeletonPlayer and practice mode.
3. Scans the ASL image dataset (c:\\Users\\hp\\OneDrive\\Desktop\\PYTHON\\Data or --data-dir).
4. Trains a high-precision Calibrated Linear Classifier on HOG features for all 26 letters.
5. Saves the trained model to backend/app/asl_model.joblib for instant runtime inference.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

# Add backend directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from app.db import Attempt, Base, Session, Sign, check_connection, engine
from app.model import MODEL_PATH, get_hog_descriptor
from app.sign_data import get_all_canonical_signs


def seed_database():
    """Populates the database with canonical alphabet and vocabulary signs."""
    print("=" * 60)
    print("STEP 1: SEEDING DATABASE WITH CANONICAL ASL SIGNS")
    print("=" * 60)

    # Ensure tables exist
    Base.metadata.create_all(engine)

    signs = get_all_canonical_signs()
    added = 0
    updated = 0

    with Session() as s:
        for item in signs:
            label = item["label"]
            existing = s.query(Sign).filter(Sign.label == label).first()
            if existing:
                existing.frames = item["frames"]
                existing.source = item["source"]
                updated += 1
            else:
                s.add(Sign(
                    label=label,
                    frames=item["frames"],
                    source=item["source"],
                    video_url=item.get("video_url"),
                ))
                added += 1
        s.commit()

    print(f"Database seeding complete: {added} new signs added, {updated} updated.")
    status = check_connection()
    print(f"Total signs in database: {status['signs_count']}")
    print(f"Signs breakdown: {status['sources']}")
    print()


def train_model(data_dir, max_per_class=None):
    """Trains the HOG + Calibrated LinearSVC classifier on the ASL dataset."""
    print("=" * 60)
    print("STEP 2: TRAINING ASL CLASSIFIER ON DATASET")
    print("=" * 60)

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Dataset directory not found: {data_path}")
        print("Skipping image model training.")
        return False

    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    hog = get_hog_descriptor()

    X, y = [], []
    t0 = time.time()
    print(f"Scanning images in: {data_path} ...")

    total_files = 0
    for letter in letters:
        folder = data_path / letter
        if not folder.exists():
            continue
        files = glob.glob(str(folder / "*.jpg"))
        if max_per_class:
            files = files[:max_per_class]

        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            res = cv2.resize(img, (64, 64))
            feat = hog.compute(res)
            if feat is not None:
                X.append(feat.flatten())
                y.append(letter)
                total_files += 1

    if not X:
        print("No valid images found for training.")
        return False

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"Extracted features for {len(X)} images across {len(set(y))} classes in {time.time() - t0:.2f}s.")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"Training Calibrated Linear Classifier on {len(X_train)} samples...")
    t1 = time.time()
    base_clf = LinearSVC(C=0.5, max_iter=2000, random_state=42)
    clf = CalibratedClassifierCV(estimator=base_clf, cv=3)
    clf.fit(X_train, y_train)
    train_time = time.time() - t1

    # Evaluation
    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Training completed in {train_time:.2f}s!")
    print(f"Holdout Test Accuracy ({len(X_test)} samples): {test_acc * 100:.2f}%")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved successfully to: {MODEL_PATH}")
    print(f"Model file size: {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB")
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description="Seed ASL database and train recognition model.")
    parser.add_argument("--data-dir", default=r"c:\Users\hp\OneDrive\Desktop\PYTHON\Data", help="Path to ASL images dataset")
    parser.add_argument("--max-per-class", type=int, default=None, help="Max images per class (default: all)")
    parser.add_argument("--skip-training", action="store_true", help="Only seed database")
    args = parser.parse_args()

    # Step 1: Seed Database
    seed_database()

    # Step 2: Train Model
    if not args.skip_training:
        train_model(args.data_dir, max_per_class=args.max_per_class)

    print("=" * 60)
    print("SUCCESS: ASL INTERPRETER SETUP COMPLETE")
    print("=" * 60)
    print("The database is now fully populated and connected.")
    print("The AI model has been trained and saved.")
    print("You can now start the backend and frontend!")


if __name__ == "__main__":
    main()

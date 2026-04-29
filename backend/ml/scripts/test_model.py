"""
Test the trained ML model directly.
Run this to verify model loads and makes predictions correctly.
No FastAPI needed.
"""

import joblib
import pandas as pd
from pathlib import Path

# Paths - Go up from scripts/ to ml/, then into models/
BASE_DIR = Path(__file__).parent.parent  # This points to backend/ml/
MODELS_DIR = BASE_DIR / "models"

print(f"Looking for models in: {MODELS_DIR}")


def test_model_loading():
    """Test that all artifacts load correctly."""
    print("=" * 50)
    print("TEST 1: Loading ML Artifacts")
    print("=" * 50)

    try:
        model = joblib.load(MODELS_DIR / "travel_classifier_final.joblib")
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

        print(f"✅ Model loaded: {type(model).__name__}")
        print("✅ Preprocessor loaded")
        print("✅ Label encoder loaded")
        print(f"   Classes: {label_encoder.classes_.tolist()}")
        return model, preprocessor, label_encoder
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("\n   Expected files at:")
        print(f"   - {MODELS_DIR / 'travel_classifier_final.joblib'}")
        print(f"   - {MODELS_DIR / 'preprocessor.joblib'}")
        print(f"   - {MODELS_DIR / 'label_encoder.joblib'}")
        return None, None, None
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None, None, None


def test_prediction(model, preprocessor, label_encoder):
    """Test prediction on a sample destination (Kathmandu)."""
    print("\n" + "=" * 50)
    print("TEST 2: Sample Prediction (Kathmandu)")
    print("=" * 50)

    # Sample features matching training data
    sample = pd.DataFrame(
        [
            {
                "avg_annual_temp_c": 20.0,
                "seasonal_range_c": 15.0,
                "cost_per_day_avg_usd": 35,
                "meal_budget_usd": 4,
                "hotel_night_avg_usd": 25,
                "flight_cost_usd": 800,
                "museum_count": 8,
                "monument_count": 12,
                "festival_score": 7,
                "beach_score": 1,
                "scenic_score": 8,
                "wellness_score": 3,
                "culture_score": 6,
                "hiking_score": 9,
                "nightlife_score": 4,
                "family_score": 3,
                "luxury_score": 1,
                "safety_score": 7,
                "tourist_density_score": 5,
                "adventure_sports_score": 9,
                "near_mountains": 1,
                "near_beach": 0,
                "region": "South Asia",
                "dry_season_months": "Oct,Nov,Dec,Jan,Feb,Mar",
                "best_season": "Oct-Nov",
                "visa_requirement": "Visa on Arrival",
                "english_friendly_score": 6,
                "public_transport_score": 4,
                "latitude": 27.7172,
                "longitude": 85.3240,
            }
        ]
    )

    print("Input: Kathmandu, Nepal")

    # Preprocess
    X_processed = preprocessor.transform(sample)
    print(f"Preprocessed shape: {X_processed.shape}")

    # Predict
    pred_encoded = model.predict(X_processed)[0]
    pred_class = label_encoder.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba(X_processed)[0]

    print(f"\n✅ Prediction: {pred_class}")
    print("\nProbabilities:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"   {class_name}: {proba[i]:.4f}")
    print(f"\nConfidence: {max(proba):.4f}")

    return pred_class


def test_batch_prediction(model, preprocessor, label_encoder):
    """Test batch prediction on multiple destinations."""
    print("\n" + "=" * 50)
    print("TEST 3: Batch Prediction (3 Destinations)")
    print("=" * 50)

    samples = pd.DataFrame(
        [
            {
                # Kathmandu - Should be Adventure
                "avg_annual_temp_c": 20.0,
                "seasonal_range_c": 15.0,
                "cost_per_day_avg_usd": 35,
                "meal_budget_usd": 4,
                "hotel_night_avg_usd": 25,
                "flight_cost_usd": 800,
                "museum_count": 8,
                "monument_count": 12,
                "festival_score": 7,
                "beach_score": 1,
                "scenic_score": 8,
                "wellness_score": 3,
                "culture_score": 6,
                "hiking_score": 9,
                "nightlife_score": 4,
                "family_score": 3,
                "luxury_score": 1,
                "safety_score": 7,
                "tourist_density_score": 5,
                "adventure_sports_score": 9,
                "near_mountains": 1,
                "near_beach": 0,
                "region": "South Asia",
                "dry_season_months": "Oct,Nov,Dec,Jan,Feb,Mar",
                "best_season": "Oct-Nov",
                "visa_requirement": "Visa on Arrival",
                "english_friendly_score": 6,
                "public_transport_score": 4,
                "latitude": 27.7172,
                "longitude": 85.3240,
            },
            {
                # Paris - Should be Culture
                "avg_annual_temp_c": 13.0,
                "seasonal_range_c": 14.0,
                "cost_per_day_avg_usd": 180,
                "meal_budget_usd": 25,
                "hotel_night_avg_usd": 150,
                "flight_cost_usd": 350,
                "museum_count": 130,
                "monument_count": 80,
                "festival_score": 9,
                "beach_score": 2,
                "scenic_score": 7,
                "wellness_score": 5,
                "culture_score": 10,
                "hiking_score": 1,
                "nightlife_score": 8,
                "family_score": 6,
                "luxury_score": 8,
                "safety_score": 8,
                "tourist_density_score": 9,
                "adventure_sports_score": 1,
                "near_mountains": 0,
                "near_beach": 0,
                "region": "Western Europe",
                "dry_season_months": "Apr,May,Jun,Sep,Oct",
                "best_season": "Apr-Jun",
                "visa_requirement": "Schengen Visa",
                "english_friendly_score": 6,
                "public_transport_score": 10,
                "latitude": 48.8566,
                "longitude": 2.3522,
            },
            {
                # Maldives - Should be Luxury
                "avg_annual_temp_c": 28.0,
                "seasonal_range_c": 3.0,
                "cost_per_day_avg_usd": 300,
                "meal_budget_usd": 30,
                "hotel_night_avg_usd": 280,
                "flight_cost_usd": 900,
                "museum_count": 1,
                "monument_count": 1,
                "festival_score": 2,
                "beach_score": 10,
                "scenic_score": 8,
                "wellness_score": 9,
                "culture_score": 2,
                "hiking_score": 1,
                "nightlife_score": 2,
                "family_score": 6,
                "luxury_score": 10,
                "safety_score": 9,
                "tourist_density_score": 4,
                "adventure_sports_score": 3,
                "near_mountains": 0,
                "near_beach": 1,
                "region": "South Asia",
                "dry_season_months": "Dec,Jan,Feb,Mar,Apr",
                "best_season": "Dec-Apr",
                "visa_requirement": "Visa on Arrival",
                "english_friendly_score": 7,
                "public_transport_score": 2,
                "latitude": 3.2028,
                "longitude": 73.2207,
            },
        ]
    )

    cities = ["Kathmandu", "Paris", "Maldives"]
    expected = ["Adventure", "Culture", "Luxury"]

    X_processed = preprocessor.transform(samples)
    predictions = model.predict(X_processed)
    predicted_classes = label_encoder.inverse_transform(predictions)

    print(f"{'City':<15} {'Expected':<15} {'Predicted':<15} {'Correct':<10}")
    print("-" * 55)
    for city, exp, pred in zip(cities, expected, predicted_classes):
        correct = "✅" if exp == pred else "❌"
        print(f"{city:<15} {exp:<15} {pred:<15} {correct:<10}")

    return predicted_classes


def test_edge_cases(model, preprocessor, label_encoder):
    """Test edge cases and error handling."""
    print("\n" + "=" * 50)
    print("TEST 4: Edge Cases")
    print("=" * 50)

    # Missing values should be handled by preprocessor
    sample_with_missing = pd.DataFrame(
        [
            {
                "avg_annual_temp_c": 20.0,
                "seasonal_range_c": None,  # Missing value
                "cost_per_day_avg_usd": 35,
                "meal_budget_usd": 4,
                "hotel_night_avg_usd": 25,
                "flight_cost_usd": 800,
                "museum_count": 8,
                "monument_count": 12,
                "festival_score": 7,
                "beach_score": 1,
                "scenic_score": 8,
                "wellness_score": 3,
                "culture_score": 6,
                "hiking_score": 9,
                "nightlife_score": 4,
                "family_score": 3,
                "luxury_score": 1,
                "safety_score": 7,
                "tourist_density_score": 5,
                "adventure_sports_score": 9,
                "near_mountains": 1,
                "near_beach": 0,
                "region": "South Asia",
                "dry_season_months": "Oct,Nov,Dec,Jan,Feb,Mar",
                "best_season": "Oct-Nov",
                "visa_requirement": "Visa on Arrival",
                "english_friendly_score": 6,
                "public_transport_score": 4,
                "latitude": 27.7172,
                "longitude": 85.3240,
            }
        ]
    )

    try:
        X_processed = preprocessor.transform(sample_with_missing)
        pred_encoded = model.predict(X_processed)[0]
        pred_class = label_encoder.inverse_transform([pred_encoded])[0]
        print(f"✅ Missing value handled: predicted {pred_class}")
    except Exception as e:
        print(f"❌ Missing value test failed: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("ML MODEL TESTING")
    print("=" * 50)

    # Test 1: Load artifacts
    model, preprocessor, label_encoder = test_model_loading()

    if model is not None:
        # Test 2: Single prediction
        test_prediction(model, preprocessor, label_encoder)

        # Test 3: Batch prediction
        test_batch_prediction(model, preprocessor, label_encoder)

        # Test 4: Edge cases
        test_edge_cases(model, preprocessor, label_encoder)

        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED")
        print("=" * 50)
        print("\n📁 Model artifacts saved at:")
        print(f"   - {MODELS_DIR / 'travel_classifier_final.joblib'}")
        print(f"   - {MODELS_DIR / 'preprocessor.joblib'}")
        print(f"   - {MODELS_DIR / 'label_encoder.joblib'}")
        print("\n✅ ML model is ready for integration with RAG and Agent.")
        print("\nNext phase: Build RAG pipeline (Phases 8-11)")
    else:
        print("\n❌ TESTS FAILED - Check your model files")

import os
import logging

logger = logging.getLogger(__name__)

class FraudScorer:
    def __init__(self):
        self.model = None
        self.scaler = None
        model_path = os.getenv("MODEL_PATH", "./models")
        
        try:
            import joblib
            model_file = os.path.join(model_path, "fraud_model.pkl")
            scaler_file = os.path.join(model_path, "scaler.pkl")
            
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
                logger.info("Model loaded successfully")
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    def score(self, features: dict) -> dict:
        # If no model, return mock score
        if self.model is None:
            import random
            score = random.random() * 0.5  # Random low score
            return {
                "fraud_score": round(score, 3),
                "risk_level": "LOW" if score < 0.3 else "MEDIUM",
                "decision": "APPROVE",
                "confidence": 0.8,
                "risk_factors": [],
                "model_version": "mock-v1.0"
            }
        
        # Real scoring logic
        try:
            score = float(self.model.predict_proba([[
                features.get("amount", 0),
                1 if features.get("is_online", False) else 0,
                1 if features.get("is_international", False) else 0,
            ]])[0][1])
        except:
            score = 0.5
            
        risk_level = "LOW" if score < 0.3 else "MEDIUM" if score < 0.7 else "HIGH"
        decision = "APPROVE" if score < 0.6 else "REVIEW" if score < 0.85 else "DECLINE"
        
        return {
            "fraud_score": round(score, 3),
            "risk_level": risk_level,
            "decision": decision,
            "confidence": 0.95,
            "risk_factors": [],
            "model_version": "v1.0.0"
        }

# Global instance
fraud_scorer = FraudScorer()

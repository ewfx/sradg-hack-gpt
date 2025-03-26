import os
import numpy as np
import requests
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime
from langchain_community.llms import LlamaCpp
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
import json
from operator import itemgetter
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self):
       # Initialize LLM
        # Download model if not already present
        MODEL_PATH = os.getenv("MODEL_PATH")
        MODEL_URL = os.getenv("MODEL_URL")
        
        if not os.path.exists(MODEL_PATH):
            logger.info(f"Downloading model from {MODEL_URL}")
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Model download complete")
            
        self.llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.1,
            max_tokens=2048,
            n_ctx=16384,
            n_threads=4,
            n_batch=512,
            verbose=False
        )
        
        # Initialize statistical model
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Define reasoning prompt
        self.reasoning_prompt = PromptTemplate(
            input_variables=["record", "statistical_score", "break_categories", "key_columns"],
            template="""Analyze record for anomalies:
Record: {record}
Score: {statistical_score}
Categories: {break_categories}
Columns: {key_columns}

Provide JSON response:
{{
    "anomalies": [
        {{
            "type": "type",
            "confidence": "0.XX",
            "reasoning": "brief explanation",
            "impact": "impact"
        }}
    ],
    "overall_assessment": "brief summary"
}}"""
        )
        
        # Create the runnable chain
        self.chain = (
            self.reasoning_prompt 
            | self.llm 
            | StrOutputParser()
        )

    def _prepare_numerical_features(self, record: Dict[str, Any]) -> np.ndarray:
        """Extract and normalize numerical features from record"""
        numerical_values = []
        for value in record.values():
            try:
                if isinstance(value, (int, float)):
                    numerical_values.append(float(value))
                elif isinstance(value, str) and value.replace(".", "").isdigit():
                    numerical_values.append(float(value))
                else:
                    numerical_values.append(0.0)
            except (ValueError, TypeError):
                numerical_values.append(0.0)
        
        return np.array(numerical_values).reshape(1, -1)

    def detect_anomalies(
        self, 
        record: Dict[str, Any],
        break_categories: List[str],
        key_columns: List[str]
    ) -> Dict[str, Any]:
        """Enhanced anomaly detection with chain of thought reasoning"""
        try:
            logger.info(f"Starting anomaly detection for record: {record}")
            
            # Step 1: Statistical Analysis
            numerical_features = self._prepare_numerical_features(record)
            statistical_score = self.isolation_forest.fit_predict(numerical_features)[0]
            # Convert numpy.int32/64 to Python int
            statistical_score = int(statistical_score)
            
            # Step 2: LLM-based Reasoning
            reasoning_input = {
                "record": json.dumps(record, indent=2),
                "statistical_score": str(statistical_score),  # Convert to string
                "break_categories": json.dumps(break_categories),  # Convert to JSON string
                "key_columns": json.dumps(key_columns)  # Convert to JSON string
            }
            
            # Updated LLM analysis with better error handling
            try:
                llm_analysis = self.chain.invoke(reasoning_input)
                # Clean up the response to ensure valid JSON
                cleaned_response = llm_analysis.strip()
                if not cleaned_response.startswith('{'):
                    # Extract JSON if it's wrapped in other text
                    start_idx = cleaned_response.find('{')
                    end_idx = cleaned_response.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        cleaned_response = cleaned_response[start_idx:end_idx]
                
                llm_result = json.loads(cleaned_response)
                
                # Ensure required fields exist
                if "anomalies" not in llm_result:
                    llm_result["anomalies"] = []
                if "overall_assessment" not in llm_result:
                    llm_result["overall_assessment"] = "No assessment provided"
                
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse LLM response as JSON: {je}\nResponse: {llm_analysis}")
                llm_result = {
                    "anomalies": [],
                    "overall_assessment": "Error parsing LLM analysis"
                }
            except Exception as e:
                logger.error(f"Error in LLM analysis: {str(e)}")
                llm_result = {
                    "anomalies": [],
                    "overall_assessment": f"Error in LLM analysis: {str(e)}"
                }
            
            # Step 5: Combine All Analysis
            final_result = {
                "statistical_analysis": {
                    "score": float(statistical_score),  # Convert to float for consistency
                    "is_anomaly": bool(statistical_score == -1)  # Convert numpy.bool_ to Python bool
                },
                "llm_analysis": llm_result,
                "final_assessment": {
                    "is_anomaly": bool(  # Convert numpy.bool_ to Python bool
                        statistical_score == -1 or
                        len(llm_result.get("anomalies", [])) > 0
                    ),
                    "confidence_score": float(self._calculate_confidence_score(  # Convert to float
                        statistical_score,
                        llm_result
                    )),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"Completed anomaly detection: {final_result['final_assessment']}")
            return final_result
        
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_confidence_score(
        self,
        statistical_score: float,
        llm_result: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score for anomaly detection"""
        confidence_factors = []
        
        # Statistical confidence
        statistical_confidence = 0.7 if statistical_score == -1 else 0.3
        confidence_factors.append(statistical_confidence)
        
        # LLM confidence
        llm_anomalies = llm_result.get("anomalies", [])
        llm_confidence = min(len(llm_anomalies) * 0.15, 0.8)
        confidence_factors.append(llm_confidence)
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors)

# Create singleton instance
_detector = None

def detect_anomalies(record: Dict[str, Any], break_categories: List[str] = None, key_columns: List[str] = None) -> Dict[str, Any]:
    """
    Global function to access anomaly detection
    """
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    
    if break_categories is None:
        break_categories = ["BREAK", "MISMATCH"]
    if key_columns is None:
        key_columns = list(record.keys())
    
    return _detector.detect_anomalies(record, break_categories, key_columns)

# import numpy as np
# from sklearn.ensemble import IsolationForest
# import logging

# logger = logging.getLogger(__name__)

# def detect_anomalies(record):
#     logger.info(f"Detecting anomalies for record: {record}")
#     model = IsolationForest()
#     try:
#         data = np.array([float(value) if isinstance(value, (int, float)) else 0 for value in record.values()]).reshape(1, -1)
#         is_anomaly = model.fit_predict(data)
#         result = "Anomaly Detected" if is_anomaly[0] == -1 else "No Anomaly"
#         logger.info(f"Anomaly detection result: {result}")
#         return result
#     except Exception as e:
#         logger.error(f"Error in anomaly detection: {e}")
#         return "Error in anomaly detection"
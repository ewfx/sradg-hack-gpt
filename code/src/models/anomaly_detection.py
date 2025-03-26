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
        
        # Define reasoning prompt with explicit JSON formatting
        self.reasoning_prompt = PromptTemplate(
            input_variables=["record", "statistical_score", "break_categories", "key_columns"],
            template="""SYSTEM: You are a financial reconciliation anomaly detector. Analyze the given record and provide your assessment in a structured JSON format.

INPUT DATA:
Record: {record}
Statistical Score: {statistical_score}
Break Categories: {break_categories}
Key Columns: {key_columns}

INSTRUCTIONS:
1. Analyze the record for anomalies
2. Determine reasoning
3. Return ONLY a valid JSON object in the following format:

{{
    "anomalies": [
        {{
            "type": "specific_anomaly_type",
            "confidence": "0.XX",
            "reasoning": "clear explanation of the anomaly"
        }}
    ],
    "overall_assessment": "brief summary of findings"
}}

REQUIREMENTS:
- Use ONLY double quotes for JSON properties and values
- Provide specific, actionable reasoning
- Do not include any text outside the JSON object
- No markdown or code block markers"""
        )
        
        # Create the runnable chain
        self.chain = (
            self.reasoning_prompt 
            | self.llm 
            | StrOutputParser()
        )

    def _clean_llm_response(self, response: str) -> Dict[str, Any]:
        """Clean and parse LLM response to ensure valid JSON"""
        try:
            # Remove any markdown code block markers
            response = response.replace("```json", "").replace("```", "").strip()
            
            # Find the first { and last }
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            # Extract just the JSON part
            json_str = response[start_idx:end_idx]
            
            # Replace single quotes with double quotes
            json_str = json_str.replace("'", '"')
            
            # Handle common JSON formatting issues
            json_str = json_str.replace("None", "null")
            json_str = json_str.replace("True", "true")
            json_str = json_str.replace("False", "false")
            
            # Validate JSON by parsing
            parsed = json.loads(json_str)
            
            # Ensure required structure
            if "anomalies" not in parsed or not isinstance(parsed["anomalies"], list):
                parsed["anomalies"] = []
            if "overall_assessment" not in parsed:
                parsed["overall_assessment"] = "No assessment provided"
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}\nOriginal response: {response}")
            # Return a default structure if parsing fails
            return {
                "anomalies": [{
                    "type": "parsing_error",
                    "confidence": "1.0",
                    "reasoning": f"Failed to parse LLM response: {str(e)}",
                    "impact": "Unable to determine impact due to parsing error"
                }],
                "overall_assessment": "Error in analysis"
            }

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
            
            # Get LLM analysis with better error handling
            llm_response = self.llm.invoke(self.reasoning_prompt.format(**reasoning_input))
            logger.info(f"Raw LLM response: {llm_response}")
            
            # Clean and parse the response
            llm_result = self._clean_llm_response(llm_response)
            
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
        if llm_anomalies:
            # Average confidence from all anomalies
            anomaly_confidences = [
                float(anomaly.get("confidence", 0))
                for anomaly in llm_anomalies
            ]
            llm_confidence = sum(anomaly_confidences) / len(anomaly_confidences)
        else:
            llm_confidence = 0.0
        
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
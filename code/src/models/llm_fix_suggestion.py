import os
import requests
from langchain_community.llms import LlamaCpp
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
# from transformers import pipeline
import logging
from dotenv import load_dotenv
import json

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model URL (Choose an optimized GGUF model)
MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_URL = os.getenv("MODEL_URL")

# Download model if not present
if not os.path.exists(MODEL_PATH):
    logger.info(f"Downloading model from {MODEL_URL}...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as model_file:
        for chunk in response.iter_content(chunk_size=8192):
            model_file.write(chunk)
    logger.info("Model downloaded successfully.")

# Load model using LlamaCpp with adjusted parameters
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,  # Lower temperature for more consistent output
    max_tokens=2048,
    n_ctx=16384,
    n_threads=4,
    n_batch=512,
    verbose=False
)

# Load Hugging Face LLM model (Example: "meta-llama/Llama-2-7b-chat-hf")
# llm_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
# llm_pipeline = LlamaCpp(model_path="path/to/your/llama-2-7b.Q4_K_M.gguf", temperature=0.7, max_tokens=256)

# Define the prompt template with system instructions
fix_suggestion_prompt = PromptTemplate(
    input_variables=["key_columns", "break_category", "record", "anomaly_result"],
    template="""You are a financial reconciliation expert. Your task is to analyze a financial record and provide a fix suggestion in JSON format.

SYSTEM INSTRUCTIONS:
1. You must respond with ONLY a valid JSON object
2. Do not include any explanatory text before or after the JSON
3. Use double quotes for all strings and keys
4. Use proper JSON array syntax for implementation_steps
5. Ensure all required fields are present

INPUT DATA:
Key columns: {key_columns}
Break category: {break_category}
Record details: {record}
Anomaly analysis: {anomaly_result}

REQUIRED JSON FORMAT:
{{
    "root_cause": "Brief explanation of the issue",
    "recommended_fix": "Specific action to take",
    "implementation_steps": ["Step 1", "Step 2", "Step 3"],
    "risk_assessment": "Brief risk analysis"
}}
"""
)

# Create the chain with output parser
fix_suggestion_chain = (
    fix_suggestion_prompt 
    | llm 
    | StrOutputParser()
)

def generate_fix_suggestion(key_columns, break_category, record, anomaly_result):
    """
    API to generate reconciliation fix suggestions using Llama/Hugging Face models.
    """
    try:
        logger.info(f"Received request: {key_columns, break_category, record, anomaly_result}")

        # Handle case where anomaly_result contains an error
        if isinstance(anomaly_result, dict) and "error" in anomaly_result:
            logger.error(f"Anomaly detection error: {anomaly_result['error']}")
            return {
                "root_cause": "Anomaly detection failed",
                "recommended_fix": "Please review the anomaly detection results",
                "implementation_steps": ["Review anomaly detection logs", "Retry anomaly detection"],
                "risk_assessment": "Unable to assess risks due to anomaly detection failure"
            }

        # Prepare input for the chain
        chain_input = {
            "key_columns": key_columns,  # Don't JSON encode these as they're already strings
            "break_category": break_category,
            "record": json.dumps(record, indent=2),
            "anomaly_result": json.dumps(anomaly_result, indent=2)
        }

        # Generate response using the chain
        response = fix_suggestion_chain.invoke(chain_input)
        logger.info(f"Raw LLM response: {response}")
        
        # Clean and parse response
        try:
            # Clean the response
            cleaned_response = response.strip()
            
            # Remove any markdown code block markers if present
            cleaned_response = cleaned_response.replace('```json', '').replace('```', '')
            
            # Find the first { and last }
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error(f"No JSON object found in response: {cleaned_response}")
                return {}
            
            # Extract just the JSON part
            json_str = cleaned_response[start_idx:end_idx]
            
            # Try to parse the JSON
            try:
                fix_suggestion = json.loads(json_str)
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error: {je}\nJSON string: {json_str}")
                # Try to fix common JSON issues
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                json_str = json_str.replace("None", "null")  # Replace Python None with JSON null
                json_str = json_str.replace("True", "true")  # Replace Python True with JSON true
                json_str = json_str.replace("False", "false")  # Replace Python False with JSON false
                fix_suggestion = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["root_cause", "recommended_fix", "implementation_steps", "risk_assessment"]
            for field in required_fields:
                if field not in fix_suggestion:
                    fix_suggestion[field] = f"Missing {field}"
                elif field == "implementation_steps" and not isinstance(fix_suggestion[field], list):
                    fix_suggestion[field] = ["Invalid implementation steps format"]
            
            logger.info(f"Generated Fix Suggestion: {fix_suggestion}")
            return fix_suggestion
            
        except json.JSONDecodeError as je:
            logger.error(f"Error parsing LLM response: {je}\nResponse: {response}")
            return {}
        
    except Exception as e:
        logger.error(f"Error generating fix suggestion: {e}")
        return {}

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from typing import Dict, Any, List
import requests
import json
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixExecutorTools:
    @staticmethod
    def update_source_system(record: Dict[str, Any], system_name: str) -> Dict[str, Any]:
        """Simulates updating source systems via API calls"""
        logger.info(f"Updating source system {system_name} with record: {record}")
        # Simulate API call
        return {
            "status": "success",
            "message": f"Updated {system_name} successfully",
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def create_jira_ticket(summary: str, description: str) -> Dict[str, Any]:
        """Simulates JIRA ticket creation"""
        ticket = {
            "ticket_id": f"RECON-{hash(summary) % 1000}",
            "summary": summary,
            "description": description,
            "status": "Open",
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Created JIRA ticket: {ticket}")
        return ticket

    @staticmethod
    def send_email_notification(recipient: str, subject: str, body: str) -> Dict[str, Any]:
        """Simulates sending email notifications"""
        email = {
            "to": recipient,
            "subject": subject,
            "body": body,
            "sent_at": datetime.now().isoformat()
        }
        logger.info(f"Sent email notification: {email}")
        return email

class FixExecutor:
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

        # Initialize fix suggestion prompt
        self.fix_prompt = PromptTemplate(
            input_variables=["record", "difference", "gl_balance", "ihub_balance"],
            template="""SYSTEM: You are a financial reconciliation expert. Analyze the record and suggest appropriate fix action.

INPUT DATA:
Record: {record}
Difference: {difference}
GL Balance: {gl_balance}
IHub Balance: {ihub_balance}

INSTRUCTIONS:
1. Analyze the record for anomalies
2. Determine appropriate fix action
3. Return ONLY a valid JSON object in the following format:

{{
    "action_type": "email_notification|jira_ticket|system_update",
    "priority": "high|medium|low",
    "description": "detailed explanation",
    "implementation": {{
        "tool": "which tool to use",
        "params": {{
            "param1": "value1",
            "param2": "value2"
        }}
    }}
}}

REQUIREMENTS:
- Use ONLY double quotes for JSON properties and values
- Do not include any text outside the JSON object
- No markdown or code block markers"""
        )

        # Initialize tools
        self.tools = [
            Tool(
                name="update_source_system",
                func=FixExecutorTools.update_source_system,
                description="Updates source systems with corrected data"
            ),
            Tool(
                name="create_jira_ticket",
                func=FixExecutorTools.create_jira_ticket,
                description="Creates JIRA tickets for tracking and documentation"
            ),
            Tool(
                name="send_email_notification",
                func=FixExecutorTools.send_email_notification,
                description="Sends email notifications to stakeholders"
            )
        ]

        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history")

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
            if "action_type" not in parsed:
                parsed["action_type"] = "jira_ticket"
            if "priority" not in parsed:
                parsed["priority"] = "medium"
            if "description" not in parsed:
                parsed["description"] = "No description provided"
            if "implementation" not in parsed:
                parsed["implementation"] = {
                    "tool": "create_jira_ticket",
                    "params": {
                        "summary": "Manual Investigation Required",
                        "description": "System failed to determine specific fix"
                    }
                }
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}\nOriginal response: {response}")
            return self._get_default_fix_action({})

    def _determine_fix_action(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Determines the appropriate fix action based on the record"""
        try:
            # Extract relevant values
            gl_balance = float(record.get('GL_Balance', 0))
            ihub_balance = float(record.get('Ihub_Balance', 0))
            difference = abs(gl_balance - ihub_balance)

            # Get fix suggestion from LLM
            prompt_input = {
                "record": json.dumps(record, indent=2),
                "difference": difference,
                "gl_balance": gl_balance,
                "ihub_balance": ihub_balance
            }
            
            response = self.llm.invoke(self.fix_prompt.format(**prompt_input))
            logger.info(f"Raw LLM response: {response}")
            
            # Clean and parse the response
            fix_action = self._clean_llm_response(response)
            logger.info(f"Determined fix action: {fix_action}")
            return fix_action
            
        except Exception as e:
            logger.error(f"Error determining fix action: {e}")
            return self._get_default_fix_action(record)

    def _get_default_fix_action(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Provides a default fix action when LLM fails"""
        return {
            "action_type": "jira_ticket",
            "priority": "medium",
            "description": "System failed to determine specific fix. Manual investigation required.",
            "implementation": {
                "tool": "create_jira_ticket",
                "params": {
                    "summary": f"Manual Investigation Required - {record.get('Account', 'Unknown')}",
                    "description": f"Failed to determine automated fix for record: {json.dumps(record, indent=2)}"
                }
            }
        }

    def execute_fix(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the fix using agentic approach"""
        logger.info(f"Starting fix execution for record: {record}")
        
        try:
            # Determine fix action
            fix_action = self._determine_fix_action(record)
            
            # Execute the determined action
            if fix_action["action_type"] == "email_notification":
                result = FixExecutorTools.send_email_notification(
                    recipient=f"{record.get('Primary_Account', 'unknown').lower()}_team@company.com",
                    subject=f"Missing Balance Alert - {record.get('Account', 'Unknown')}",
                    body=f"Missing balance detected in reconciliation:\n\n{json.dumps(record, indent=2)}"
                )
            elif fix_action["action_type"] == "jira_ticket":
                result = FixExecutorTools.create_jira_ticket(
                    summary=fix_action["implementation"]["params"]["summary"],
                    description=fix_action["implementation"]["params"]["description"]
                )
            elif fix_action["action_type"] == "system_update":
                result = FixExecutorTools.update_source_system(
                    record=record,
                    system_name="ihub" if record.get('GL_Balance', 0) != 0 else "gl"
                )
            else:
                result = self._get_default_fix_action(record)

            return {
                "status": "success",
                "action_taken": fix_action["action_type"],
                "priority": fix_action["priority"],
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Error executing fix: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "record_id": record.get("Account", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }

def execute_fix(record: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for fix execution"""
    executor = FixExecutor()
    return executor.execute_fix(record)

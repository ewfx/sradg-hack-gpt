from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
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
            n_ctx=16384,
            max_tokens=2048,
            n_threads=4,
            n_batch=512,
            verbose=False
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

    def _determine_fix_strategy(self, record: Dict[str, Any], anomaly_result: str) -> List[str]:
        """Determines the sequence of actions needed to fix the issue"""
        prompt = f"""
        Analyze the following reconciliation record and anomaly:
        Record: {json.dumps(record)}
        Anomaly: {anomaly_result}

        Determine the necessary steps to resolve this issue. Consider:
        1. Which source systems need updates
        2. Whether stakeholder notification is needed
        3. If a JIRA ticket should be created for tracking

        Return the list of required actions.
        """
        
        response = self.llm(prompt)
        # Parse response to get action steps
        actions = [step.strip() for step in response.split('\n') if step.strip()]
        return actions

    def execute_fix(self, record: Dict[str, Any], anomaly_result: str = None) -> Dict[str, Any]:
        """
        Executes the fix using agentic approach
        """
        logger.info(f"Starting fix execution for record: {record}")
        
        try:
            # Determine fix strategy
            fix_actions = self._determine_fix_strategy(record, anomaly_result)
            execution_results = []

            for action in fix_actions:
                if "update" in action.lower():
                    # Execute source system updates
                    for system in ["source1", "source2"]:
                        result = FixExecutorTools.update_source_system(record, system)
                        execution_results.append({
                            "action": f"update_{system}",
                            "result": result
                        })

                if "jira" in action.lower():
                    # Create JIRA ticket
                    ticket_result = FixExecutorTools.create_jira_ticket(
                        summary=f"Reconciliation Fix for {record.get('Transaction ID', 'Unknown')}",
                        description=f"Automated fix executed for record:\n{json.dumps(record, indent=2)}\n\nAnomaly: {anomaly_result}"
                    )
                    execution_results.append({
                        "action": "create_jira_ticket",
                        "result": ticket_result
                    })

                if "email" in action.lower() or "notify" in action.lower():
                    # Send notification
                    email_result = FixExecutorTools.send_email_notification(
                        recipient="stakeholder@company.com",
                        subject=f"Reconciliation Fix Executed - {record.get('Transaction ID', 'Unknown')}",
                        body=f"Automated fix has been executed for the following record:\n\n{json.dumps(record, indent=2)}"
                    )
                    execution_results.append({
                        "action": "send_notification",
                        "result": email_result
                    })

            # Compile execution summary
            summary = {
                "status": "success",
                "record_id": record.get("Transaction ID", "Unknown"),
                "actions_executed": len(execution_results),
                "execution_details": execution_results,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Fix execution completed successfully: {summary}")
            return summary

        except Exception as e:
            error_msg = f"Error executing fix: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "record_id": record.get("Transaction ID", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }

def execute_fix(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for fix execution
    """
    executor = FixExecutor()
    return executor.execute_fix(record)

import pandas as pd
import requests
import json
import logging
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFormatter:
    def __init__(self):
        self.openrefine_url = os.getenv("OPENREFINE_URL", "http://localhost:3333")
        self.project_name = "reconciliation_data"
        
    def create_project(self, df: pd.DataFrame) -> str:
        """Create a new OpenRefine project with the data"""
        try:
            # Convert DataFrame to CSV string
            csv_data = df.to_csv(index=False)
            
            # Create project using OpenRefine API
            response = requests.post(
                f"{self.openrefine_url}/command/core/create-project",
                files={
                    "project-file": ("data.csv", csv_data, "text/csv")
                },
                data={
                    "project-name": self.project_name
                }
            )
            
            if response.status_code == 200:
                project_id = response.json().get("project_id")
                logger.info(f"Created OpenRefine project with ID: {project_id}")
                return project_id
            else:
                raise Exception(f"Failed to create project: {response.text}")
                
        except Exception as e:
            logger.error(f"Error creating OpenRefine project: {str(e)}")
            raise

    def apply_transformations(self, project_id: str, transformations: List[Dict[str, Any]]) -> None:
        """Apply data transformations using OpenRefine"""
        try:
            for transform in transformations:
                # Apply transformation using OpenRefine API
                response = requests.post(
                    f"{self.openrefine_url}/command/core/apply-operations",
                    json={
                        "project": project_id,
                        "operations": [transform]
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Failed to apply transformation: {response.text}")
                    
            logger.info("Successfully applied all transformations")
            
        except Exception as e:
            logger.error(f"Error applying transformations: {str(e)}")
            raise

    def export_data(self, project_id: str) -> pd.DataFrame:
        """Export formatted data from OpenRefine"""
        try:
            # Export data as CSV
            response = requests.get(
                f"{self.openrefine_url}/command/core/export-rows",
                params={
                    "project": project_id,
                    "format": "csv"
                }
            )
            
            if response.status_code == 200:
                # Convert response to DataFrame
                df = pd.read_csv(pd.StringIO(response.text))
                logger.info(f"Successfully exported {len(df)} rows")
                return df
            else:
                raise Exception(f"Failed to export data: {response.text}")
                
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify numeric columns in the DataFrame"""
        numeric_columns = []
        for column in df.columns:
            # Check if column contains numeric data
            if df[column].dtype in ['int64', 'float64'] or df[column].str.replace('.', '').str.isnumeric().all():
                numeric_columns.append(column)
        return numeric_columns

    def _get_balance_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify balance columns based on common naming patterns"""
        balance_patterns = ['balance', 'amount', 'value', 'total', 'sum']
        balance_columns = []
        
        for column in df.columns:
            column_lower = column.lower()
            if any(pattern in column_lower for pattern in balance_patterns):
                balance_columns.append(column)
        
        return balance_columns

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to format data using OpenRefine"""
        try:
            # Create OpenRefine project
            project_id = self.create_project(df)
            
            # Get numeric and balance columns
            numeric_columns = self._get_numeric_columns(df)
            balance_columns = self._get_balance_columns(df)
            
            # Define common transformations
            transformations = []
            
            # Add numeric transformations for all numeric columns
            for column in numeric_columns:
                transformations.append({
                    "op": "core/text-transform",
                    "engineConfig": {
                        "facets": [],
                        "mode": "row-based"
                    },
                    "columnName": column,
                    "expression": "value.toNumber()",
                    "onError": "keep-original",
                    "repeat": False,
                    "repeatCount": 10,
                    "description": f"Convert {column} to number"
                })
            
            # Add balance-specific transformations
            for column in balance_columns:
                transformations.extend([
                    # Convert to number
                    {
                        "op": "core/text-transform",
                        "engineConfig": {
                            "facets": [],
                            "mode": "row-based"
                        },
                        "columnName": column,
                        "expression": "value.toNumber()",
                        "onError": "keep-original",
                        "repeat": False,
                        "repeatCount": 10,
                        "description": f"Convert {column} to number"
                    },
                    # Remove currency symbols if present
                    {
                        "op": "core/text-transform",
                        "engineConfig": {
                            "facets": [],
                            "mode": "row-based"
                        },
                        "columnName": column,
                        "expression": "value.replace(/[^0-9.-]/g, '')",
                        "onError": "keep-original",
                        "repeat": False,
                        "repeatCount": 10,
                        "description": f"Remove currency symbols from {column}"
                    }
                ])
            
            # Add standard transformations for common fields
            standard_transformations = [
                # Standardize currency codes
                {
                    "op": "core/text-transform",
                    "engineConfig": {
                        "facets": [],
                        "mode": "row-based"
                    },
                    "columnName": "Currency",
                    "expression": "value.toUpperCase()",
                    "onError": "keep-original",
                    "repeat": False,
                    "repeatCount": 10,
                    "description": "Standardize currency codes"
                },
                # Clean account numbers
                {
                    "op": "core/text-transform",
                    "engineConfig": {
                        "facets": [],
                        "mode": "row-based"
                    },
                    "columnName": "Account",
                    "expression": "value.trim()",
                    "onError": "keep-original",
                    "repeat": False,
                    "repeatCount": 10,
                    "description": "Clean account numbers"
                },
                # Standardize dates
                {
                    "op": "core/text-transform",
                    "engineConfig": {
                        "facets": [],
                        "mode": "row-based"
                    },
                    "columnName": "As_of_Date",
                    "expression": "value.replace(/(\\d{1,2})\\/(\\d{1,2})\\/(\\d{4})/, '$3-$2-$1')",
                    "onError": "keep-original",
                    "repeat": False,
                    "repeatCount": 10,
                    "description": "Standardize date format"
                }
            ]
            
            # Add standard transformations if columns exist
            for transform in standard_transformations:
                if transform["columnName"] in df.columns:
                    transformations.append(transform)
            
            # Apply transformations
            self.apply_transformations(project_id, transformations)
            
            # Export formatted data
            formatted_df = self.export_data(project_id)
            
            return formatted_df
            
        except Exception as e:
            logger.error(f"Error in data formatting: {str(e)}")
            # Return original DataFrame if formatting fails
            return df

def format_reconciliation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Global function to access data formatting"""
    formatter = DataFormatter()
    return formatter.format_data(df) 
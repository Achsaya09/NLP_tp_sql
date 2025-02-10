__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st
from openai import OpenAI
from typing import Optional
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
import asyncio
from sqlalchemy.exc import SQLAlchemyError
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat

from cryptography.fernet import Fernet
import json
with open('secret.key', 'rb') as key_file:
        key = key_file.read()
 
        cipher_suite = Fernet(key)
 
        # Load the encrypted configuration data
        with open('config.json', 'r') as config_file:
            encrypted_data = json.load(config_file)
 
        # Decrypt the sensitive information
        data = {key: cipher_suite.decrypt(value.encode()).decode() for key, value in encrypted_data.items()}
        


class AthenaAgent(Workflow):
    def __init__(self, vn, db_type: str, db_credentials: dict, timeout: Optional[float] = 200.0):
        super().__init__(timeout=timeout)
        self.vn = vn
        self.db_type = db_type
        self.db_credentials = db_credentials
        self.engine = self.create_db_connection()
        self.vn.run_sql = self.run_query
        self.vn.run_sql_is_set = True

    def create_db_connection(self):
        """Create a database connection dynamically based on the selected type."""
        if self.db_type == "AWS Athena":
            aws_access_key = self.db_credentials.get("aws_access_key")
            aws_secret_key = self.db_credentials.get("aws_secret_key")
            region_name = self.db_credentials.get("region_name")
            db_name = self.db_credentials.get("db_name")
            s3_output_location = self.db_credentials.get("s3_output_location")

            return create_engine(
                f'awsathena+rest://{aws_access_key}:{aws_secret_key}@athena.{region_name}.amazonaws.com/{db_name}?s3_staging_dir={s3_output_location}'
            )
        elif self.db_type == "Azure Synapse":
            server = self.db_credentials.get("server")
            database = self.db_credentials.get("database")
            username = self.db_credentials.get("username")
            password = self.db_credentials.get("password")

            return create_engine(
                f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server'
            )
        elif self.db_type == "GCP BigQuery":
            project_id = self.db_credentials.get("project_id")
            return create_engine(f'bigquery://{project_id}')

        else:
            raise ValueError("Unsupported database type")

    def run_query(self, sql: str) -> pd.DataFrame:
        """Execute a query against the database and return the result as a DataFrame."""
        with self.engine.connect() as connection:
            result = connection.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df

    def get_schema(self) -> dict:
        """Retrieve schema information for documentation and querying."""
        schema_query = """
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME IN ('outputfilevm', 'outputfilevmdisk')
        """
        df_schema = self.run_query(schema_query)

        schema = {}
        for table_name, group in df_schema.groupby('TABLE_NAME'):
            schema_details = group.to_dict(orient='records')

            sample_query = f"SELECT * FROM {table_name} LIMIT 5"
            try:
                sample_data = self.run_query(sample_query).to_dict(orient='records')
            except Exception as e:
                sample_data = f"Error fetching sample data: {e}"

            schema[table_name] = {
                "columns": schema_details,
                "sample_data": sample_data
            }
        return schema

    def engineer_prompt(self, question: str, schema: dict) -> str:
        """Generate a prompt for SQL query generation based on the schema and user question."""
        return f"""
        You are a SQL expert tasked with generating a query based on the provided schema. It is critical to adhere strictly to the schema and use the exact table and column names as specified.

**Database Name**: {self.db_credentials.get('db_name', 'N/A')}

**Schema**:
{schema}

**Task**:
1. **Objective**: Write an SQL query to address the following question:
   *"{question}"*

2. **Pre-query Requirements**:
   - **Read-only operation**: The query must only perform read operations. No modifications such as `INSERT`, `UPDATE`, or `DELETE` are permitted.
   - **Schema Validation**: Carefully review the schema to:
     - Identify the correct tables and columns relevant to the question.
     - Verify all table and column names match exactly as specified in the schema.
     - Understand the relationships between tables (if applicable).
     - **Always use `GROUP BY` for all the queries** 

3. **Query Construction Guidelines**:
   - Use only the table names, column names, and relationships explicitly provided in the schema.
   - Do not assume or invent any table or column names that are not in the schema.
   - If the schema does not contain all the necessary information for the query, provide a detailed explanation of why the query cannot be completed.

4. **Validation Checklist**:
   - Double-check that all table and column names in the query exactly match the schema.
   - Ensure that relationships between tables (if used) align with those described in the schema.
   - Confirm that the query fully addresses the question within the constraints of the schema.

5. **Output**:
   - If the query can be written, return only the SQL query.
   - If the query cannot be generated due to insufficient or unclear schema information, provide a detailed rationale for why it is not possible.

**Note**: Adherence to the schema is mandatory. Queries that do not align with the schema or include assumptions will be considered invalid.
Return only sql query
        """

    @step
    async def start_chat(self, ev: StartEvent) -> StopEvent:
        question = ev.topic
        try:
            schema = self.get_schema()
            prompt = self.engineer_prompt(question, schema)

            sql_query = self.vn.generate_sql(prompt)
            result_df = self.run_query(sql_query)

            plotly_code = self.vn.generate_plotly_code(question=question, sql=sql_query, df=result_df)
            fig = self.vn.get_plotly_figure(plotly_code=plotly_code, df=result_df)

            return StopEvent(result=[result_df, fig])
        except Exception as e:
            st.error(f"Error: {e}")
            return StopEvent(result="An error occurred while processing your query. Please try again.")

def main():
    class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)

    st.title("Dynamic Query Interface")
    st.write("Select a database, enter credentials, and ask your questions!")

    db_type = st.selectbox("Choose Database Type", ["AWS Athena", "Azure Synapse", "GCP BigQuery"])

    db_credentials = {}
    if db_type == "AWS Athena":
        db_credentials["aws_access_key"] = st.text_input("AWS Access Key")
        db_credentials["aws_secret_key"] = st.text_input("AWS Secret Key", type="password")
        db_credentials["region_name"] = st.text_input("AWS Region")
        db_credentials["db_name"] = st.text_input("Database Name")
        db_credentials["s3_output_location"] = st.text_input("S3 Output Location")
    elif db_type == "Azure Synapse":
        db_credentials["server"] = st.text_input("Server")
        db_credentials["database"] = st.text_input("Database Name")
        db_credentials["username"] = st.text_input("Username")
        db_credentials["password"] = st.text_input("Password", type="password")
    elif db_type == "GCP BigQuery":
        db_credentials["project_id"] = st.text_input("Project ID")

    if st.button("Connect"):
        vn = MyVanna(config={'api_key': data["API_KEY"], 'model': 'gpt-3.5-turbo', 'temperature': 0.2,'path': 'embeddings_dir_dynamic_azure'})
        athena_agent = AthenaAgent(vn=vn, db_type=db_type, db_credentials=db_credentials)
        st.session_state["athena_agent"] = athena_agent

    if "athena_agent" in st.session_state:
        athena_agent = st.session_state["athena_agent"]

        user_input = st.text_input("Enter your question (or leave blank to exit):")

        if user_input:
            async def process_input():
                start_event = StartEvent(topic=user_input)
                stop_event = await athena_agent.start_chat(start_event)

                if stop_event != None:
                    if isinstance(stop_event.result, list):
                        result_df, fig = stop_event.result
                        st.write("### Query Result:")
                        st.dataframe(result_df)

                        st.write("### Generated Plot:")
                        st.plotly_chart(fig)
                    else:
                        st.write("### Result:")
                        st.write(stop_event.result)
                else:
                    st.write("No result available. Please try again.")

            asyncio.run(process_input())

if __name__ == "__main__":
    main()


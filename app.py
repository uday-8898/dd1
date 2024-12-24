from openai import AzureOpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from model import extract_content_based_on_query
import json
from fastapi import Query,Body
import requests
from io import BytesIO
import pdfkit
import re
from fastapi.responses import JSONResponse
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import AzureChatOpenAI
import pandas as pd
 
from fastapi.responses import FileResponse
from typing import Dict
from azure.core.exceptions import HttpResponseError
 
from rag_data_processing import convert_excel_to_pdf,extact_content_embedding_from_file,extact_content_embedding_from_csv_file,extract_text_from_pdf,read_and_split_pdf,extact_content_embedding_from_word_file,read_and_split_word,upload_files_to_blob
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import os
from rag_data_processing import CONNECTION_STRING, CONTAINER_NAME
from pydantic import BaseModel
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
import shutil
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from history import *
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
from mysql.connector import Error
import random
import string
from model import  extract_content_based_on_query,process_file,process_and_upload_embeddings
from middleware import expiry_check

 
# Azure OpenAI Configuration (directly defined in code)
AZURE_OPENAI_API_KEY = "24d6f09f8b0f44b1a1b90da2488fb417"
AZURE_OPENAI_ENDPOINT = "https://openai-production001.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4-32k-cbt"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"# Add the API version here
#update
# Connection details
host_name = "mysqlai.mysql.database.azure.com"
user_name = "azureadmin"
user_password = "Meridian@123"
db_name = "chatbot"


db_config = {
    "host": host_name,
    "database": db_name,
    "user": user_name,
    "password": user_password
}
 
 
chat_client = AzureOpenAI(
  azure_endpoint = "https://openai-production001.openai.azure.com/openai/deployments/gpt-4-km/chat/completions?api-version=2024-08-01-preview",
  api_key="24d6f09f8b0f44b1a1b90da2488fb417",  
  api_version="2024-08-01-preview"
)
 
def get_response_from_query(query, content, history, language):
    message = [
        {"role": "system", "content": f"You are an AI assistant that helps to answer the questions from the given content in {language} language. Give the response in JSON."},
        {"role": "user", "content": f"""Your task is to follow chain of thought method to first extract accurate answer for given user query, chat history and provided input content. Then change the language of response into {language} language. Give the response in the json format only having 'bot answer' and 'scope' as key.\n\nInput Content : {content} \n\nUser Query : {query}\n\nChat History : {history}\n\nImportant Points while generating response:\n1. The answer of the question should be relevant to the input text.\n2. Answer complexity would be based on input content.\n3. If input content is not provided direct the user to provide content.\n4. Answers should not be harmful or spam. If there is such content give the instructions to user accordingly. \n5. If user query is out of scope of given content give the value of 'scope' key False.\n6. Give the response in the json format. \n\nExtracted json response:"""}
    ]
 
    response = chat_client.chat.completions.create(
      model="gpt-4-km", # model = "deployment_name"
      messages = message,
      temperature=0.7,
      max_tokens=1000,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None,
    )
    # Loading the response as a JSON object
    json_response = json.loads(response.choices[0].message.content)
    print(json_response)
    return json_response

def language_correct_query(query, history):
    # Prepare the history messages in the format expected by the model
    formatted_history = []
    for entry in history:
        # Assuming the history contains 'bot_answer' and 'citation_dict'
        # and you want to use the 'bot_answer' as the response content
        if 'bot_answer' in entry:
            formatted_history.append({"role": "assistant", "content": entry['bot_answer']})
        if 'user_query' in entry:
            formatted_history.append({"role": "user", "content": entry['user_query']})

    # Define the message for the model
    message = [
        {"role": "system", "content": "You are an AI assistant that helps to identify and extract the language, fix typing errors, and translate any non-English content into English based on the user's query."},
        {"role": "user", "content": f"""Your task is to identify and extract the language of the query string, fix typing errors, and translate any content into English if it's not in English. Always return the response in JSON format.\n\nInput Content: {query}\n\nHistory: {formatted_history}\n\nImportant Instructions:\n1. Identify the language of the content (e.g., English, French, etc.).\n2. Correct any typing errors and translate the content into English if it is not already in English.\n3. If the content is already in English and does not need modification, just return the content as is.\n4. Provide a response in JSON format with 'Language' and 'Modified Content'.\n\nExtracted JSON Response:"""}
    ]

    # Make the API call to OpenAI
    response = chat_client.chat.completions.create(
        model="gpt-4-km",  # model = "deployment_name"
        messages=message,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )

    # Loading the response as a JSON object
    json_response = json.loads(response.choices[0].message.content)
    return json_response

def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string
 
 
 
# Define the query request model
class QueryRequest(BaseModel):
    query : str
    database : str
    email : str
 
class DownloadRequest(BaseModel):
    folder_name: str
 
def background_task(folder_path: str):

    _ = extact_content_embedding_from_file(folder_path)
    print(f"Background task completed ")

def background_task3(folder_path: str):
    _ = extact_content_embedding_from_csv_file(folder_path)
    print(f"Background task completed ")

def background_task2(folder_path: str):
    # Simulate a long-running task
    _ = extact_content_embedding_from_word_file(folder_path)
    print(f"Background task completed ")
# Define the response model
class QueryResponse(BaseModel):
    bot_answer: str
    citation_dict: list
class QueryResponsesql(BaseModel):
    bot_answer: str
 
# Pydantic model for request validation
class UserRegistration(BaseModel):
    name: str
    email: str
 
def download_blobs_from_folder(container_name, folder_name, connection_string, local_download_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    folder_path = os.path.join(local_download_path, folder_name)
    print(folder_path)
   
    # Create local download path if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
   
    blob_list = container_client.list_blobs(name_starts_with=folder_name)
    csv_blobs = [blob for blob in blob_list if blob.name.endswith('.csv')]
   
    if not csv_blobs:
        print("No .csv files found in the folder.")
        return False
 
    for blob in csv_blobs:
        blob_client = container_client.get_blob_client(blob.name)
        local_file_path = os.path.join(folder_path, os.path.relpath(blob.name, folder_name))
       
        # Create directories if they don't exist
        local_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
       
        with open(local_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        print(f"Downloaded {blob.name} to {local_file_path}")
   
    return True



 
def download_blobs_from_folder_BC(container_name, folder_name, connection_string, local_download_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    folder_path = os.path.join(local_download_path, folder_name)
    print(folder_path)
   
    # Create local download path if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
   
    # List blobs and filter for .csv and .xlsx files
    blob_list = container_client.list_blobs(name_starts_with=folder_name)
    data_blobs = [blob for blob in blob_list if blob.name.endswith('.csv') or blob.name.endswith('.xlsx') or blob.name.endswith('.xls')]
   
    if not data_blobs:
        print("No .csv or .xlsx files found in the folder.")
        return False
 
    for blob in data_blobs:
        blob_client = container_client.get_blob_client(blob.name)
        local_file_path = os.path.join(folder_path, os.path.relpath(blob.name, folder_name))
       
        # Create directories if they don't exist
        local_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
       
        with open(local_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        print(f"Downloaded {blob.name} to {local_file_path}")
   
    return True

 
 
async def respond_to_question(original_query_string, folder_name, email_id):
    print("432")
    current_working_directory = os.getcwd()
    db_path = os.path.join(current_working_directory, folder_name)
    print("432")
    print(db_path)
    if not os.path.exists(db_path):
        result = download_blobs_from_folder(CONTAINER_NAME, folder_name, CONNECTION_STRING, current_working_directory)
        if result ==True:
            print("the blob is fetched")
            print("432")
        if result == False:
            return {"bot_answer": "Data Base not created yet", "citation_dict": {}}
        print("432")

    history = extract_and_format_last_two_records(folder_name, email_id)

# Assign default value if history is empty
    if not history:
        history = [
        {"role": "assistant", "content": "No previous chat history found."}
    ]
    else:
    # Assuming `history` is a list of dicts with "bot_answer" and "citation_dict"
    # You would need to transform it into the format required by the function
        history = [
        {"role": "assistant", "content": record["bot_answer"]}
        for record in history
    ]   
    
    print(history)    
    # This function should already exist with the required logic
    language_response = language_correct_query(original_query_string, history)
    # Placeholder response logic
    query_string = language_response["Modified Content"]
    content_list, citation_dict = extract_content_based_on_query(query_string, 10,folder_name)
    print("content fetch successfully")
    content = " ".join(content_list)
    
    answer = get_response_from_query(query_string, content, history, language_response["Language"].strip().lower())
    if answer["scope"] == False:
        citation_dict = []
    output_response = {"bot_answer": answer["bot answer"], "citation_dict": citation_dict}  
    db_response = [{"user_question": original_query_string, "answer" : output_response }]
    user_id = get_user_id_by_email(email_id)    
    print("user_id",user_id)
    store_data_in_db(query_string,output_response['bot_answer'], user_id, folder_name)
    # append_data(history_data_path, output_response)
    print(output_response)
    return output_response
 

def get_user_id_by_email(email_id):
    # Database connection configuration
    db_config = {
    "host": host_name,
    "database": db_name,
    "user": user_name,
    "password": user_password
}
 
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)
 
        if connection.is_connected():
            cursor = connection.cursor()
            # Query to fetch user_id based on email_id
            fetch_user_id_query = """
            SELECT user_id FROM km_registration WHERE email_id = %s
            """
            cursor.execute(fetch_user_id_query, (email_id,))
            result = cursor.fetchone()
 
            if result:
                user_id = result[0]
                return user_id
            else:
                return None
 
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None
 
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
 
def get_file_type_by_database(folder_name):
    # Database connection configuration
    db_config = {
        "host": host_name,
        "database": db_name,
        "user": user_name,
        "password": user_password
    }
    
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)
 
        if connection.is_connected():
            cursor = connection.cursor()
            # Query to fetch folder_type based on folder_name
            fetch_user_id_query = """
            SELECT folder_type FROM km_db_mapping WHERE db_name = %s;
            """
            # Pass folder_name as a tuple
            cursor.execute(fetch_user_id_query, (folder_name,))
            result = cursor.fetchone() 
            if result:
                folder_type = result[0]
                return folder_type
            else:
                return None
 
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None
 
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


            
def check_folder_exists(user_id, folder_name):
    # Database connection configuration
    db_config = {
        "host": host_name,
        "database": db_name,
        "user": user_name,
        "password": user_password
    }
    
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)
        
        if connection.is_connected():
            cursor = connection.cursor()
            # Check if the folder name already exists for the given user_id
            check_query = """
            SELECT * FROM km_db_mapping WHERE user_id = %s AND db_name = %s
            """
            cursor.execute(check_query, (user_id, folder_name))
            result = cursor.fetchone()
            return result is not None  # Returns True if folder exists, False otherwise

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return False

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
def add_db_mapping(user_id, database_name,folder_type,visibility):
    # Database connection configuration
    db_config = {
    "host": host_name,
    "database": db_name,
    "user": user_name,
    "password": user_password
}
 
 
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)
 
        if connection.is_connected():
            cursor = connection.cursor()
            # Insert the user_id and db_name into the km_db_mapping table
            insert_query = """
            INSERT INTO km_db_mapping (user_id, db_name, folder_type, visibility) VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (user_id, database_name,folder_type,visibility))
            connection.commit()
            return {"message": "Mapping added successfully", "user_id": user_id, "db_name": database_name,"folder_type":folder_type,"visibility":visibility}
 
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return {"error": str(e)}
 
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
 
 
 
def get_db_names_by_user_id(user_id):
    # Database connection configuration
    db_config = {
        "host": host_name,
        "database": db_name,
        "user": user_name,
        "password": user_password
    }

    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)

        if connection.is_connected():
            cursor = connection.cursor()
            # Query to fetch db_name, folder_type, and visibility for the given user_id
            fetch_db_names_query = """
            SELECT db_name, folder_type, visibility 
            FROM km_db_mapping 
            WHERE user_id = %s and visibility = 'local'
            """
            cursor.execute(fetch_db_names_query, (user_id,))
            results = cursor.fetchall()
            print("Raw Results:", results)

            # Construct a list of dictionaries for all fetched rows
            db_names = [
                {"db_name": row[0], "folder_type": row[1], "visibility": row[2]} 
                for row in results
            ]

            # Returning the key `db_names` instead of `data`
            return {"user_id": user_id, "db_names": db_names}

    except mysql.connector.Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return {"error": str(e)}

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()



def get_global_db_names_by_user_id():
    # Database connection configuration
    db_config = {
        "host": host_name,
        "database": db_name,
        "user": user_name,
        "password": user_password
    }

    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)
        visibility='global'
        if connection.is_connected():
            cursor = connection.cursor()
            # Query to fetch db_name, folder_type, and visibility for the given user_id
            fetch_db_names_query = """
            SELECT db_name, folder_type, visibility 
            FROM km_db_mapping 
            WHERE visibility = %s
            """
            cursor.execute(fetch_db_names_query, (visibility,))
            results = cursor.fetchall()
            print("Raw Results:", results)

            # Construct a list of dictionaries for all fetched rows
            db_names = [
                {"db_name": row[0], "folder_type": row[1], "visibility": row[2]} 
                for row in results
            ]

            # Returning the key `db_names` instead of `data`
            return {"db_names": db_names}

    except mysql.connector.Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return {"error": str(e)}

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()





import mysql.connector
from mysql.connector import Error

def delete_database_record(user_id, db_name):
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            cursor = connection.cursor()
            print("a")
            delete_query = """
            DELETE FROM km_db_mapping 
            WHERE user_id = %s AND db_name = %s
            """
            print("a")
            delete_query1 = """
            DELETE FROM km_chat_history
            WHERE user_id = %s AND db_name = %s
            """
            print("a")
            # Execute both queries separately
            cursor.execute(delete_query, (user_id, db_name))
            rows_affected_query1 = cursor.rowcount
            print("a")
            cursor.execute(delete_query1, (user_id, db_name))
            rows_affected_query2 = cursor.rowcount
            print("a")
            connection.commit()
            return True 
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return False
    finally:
        try:
            if cursor:
                cursor.close()
        except NameError:
            pass
        try:
            if connection.is_connected():
                connection.close()
        except NameError:
            pass

def delete_all_database_record(user_id):
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            cursor = connection.cursor()
            
            delete_query = """
            DELETE FROM km_db_mapping 
            WHERE user_id = %s
            """
            
            delete_query1 = """
            DELETE FROM km_chat_history 
            WHERE user_id = %s
            """
            
            # Execute the first delete query
            cursor.execute(delete_query, (user_id,))
            rows_deleted_1 = cursor.rowcount  # Rows affected by the first query
            
            # Execute the second delete query
            cursor.execute(delete_query1, (user_id,))
            rows_deleted_2 = cursor.rowcount  # Rows affected by the second query
            
            # Commit changes
            connection.commit()
            
            # Return True if any of the queries deleted rows
            return (rows_deleted_1 > 0) or (rows_deleted_2 > 0)
    
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return False
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()




db_connection = None
metadata = {}

# Hardcoded database configuration
def configure_database():
    global db_connection, metadata
    try:
        db_connection = mysql.connector.connect(
            host="mysqlai.mysql.database.azure.com",
            user="azureadmin",
            password="Meridian@123",
            database="chatbot"
        )

        if db_connection.is_connected():
            cursor = db_connection.cursor(dictionary=True)
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()

            metadata = {}
            for table in tables:
                table_name = table[f'Tables_in_chatbot']
                cursor.execute(f"DESCRIBE `{table_name}`")  # Escape table names with backticks
                columns = cursor.fetchall()
                metadata[table_name] = [column['Field'] for column in columns]

            print("Database configured successfully.")
            print("Metadata:", metadata)
        else:
            raise HTTPException(status_code=500, detail="Failed to connect to the database.")
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
def validate_query(sql_query: str) -> bool:
    """Validate the SQL query against metadata."""
    try:
        lower_query = sql_query.lower()
        for table_name, columns in metadata.items():
            if table_name.lower() in lower_query:
                # Check if any column from the metadata is mentioned in the query
                for column in columns:
                    if column.lower() in lower_query:
                        print(f"Column '{column}' found in query: {sql_query}")
                    else:
                        print(f"Column '{column}' NOT found in query: {sql_query}")
        return True
    except Exception as e:
        print(f"Error validating query: {str(e)}")
        return False

async def respond_to_question_sql(original_query_string, folder_name, email_id):
    global db_connection, metadata
    if not db_connection or not metadata:
        configure_database()

    try:
        # Step 1: Split the original query into sub-questions
        split_prompt = (
            "Analyze the following user input, identify all independent logical sub-queries, and break them into distinct SQL-like questions: "
            f"'{original_query_string}'.\n"
            "Ensure that each sub-query is logically independent and self-contained. Return each sub-query on a new line without any explanations or additional text."
        )

        split_response = chat_client.chat.completions.create(
            model="gpt-4-km",
            messages=[ 
                {"role": "system", "content": "You are a helpful assistant that extracts logical sub-questions."},
                {"role": "user", "content": split_prompt},
            ],
            max_tokens=1000  # Reduced token limit
        )

        # Extract split queries
        split_queries = split_response.choices[0].message.content.strip().split("\n")

        results = []
        full_response = ""  # Variable to store the combined response

        # Step 2: Process each query independently
        for sub_question in split_queries:
            if not sub_question.strip():
                continue

            # Generate SQL query (unchanged)
            sql_generation_prompt = (
                "You are a helpful assistant that generates SQL queries only. "
                f"Do not include any explanations or comments. Generate an SQL query for the following question:\n'{sub_question}'\n"
                "Tables and columns:\n" +
                "\n".join([f"{table}: {', '.join(columns)}" for table, columns in metadata.items()])
            )

            sql_response = chat_client.chat.completions.create(
                model="gpt-4-km",
                messages=[ 
                    {"role": "system", "content": sql_generation_prompt},
                ],
                max_tokens=1000  # Reduced token limit
            )

            raw_sql = sql_response.choices[0].message.content.strip()
            cleaned_sql = raw_sql.replace("```sql", "").replace("```", "").strip()

            if not cleaned_sql.lower().startswith("select"):
                raise HTTPException(status_code=400, detail=f"Invalid SQL query generated for '{sub_question}'.")
            print(cleaned_sql)
            # Execute the SQL query
            cursor = db_connection.cursor(dictionary=True)
            cursor.execute(cleaned_sql)
            rows = cursor.fetchall()

            # Limit rows to avoid exceeding token limit
            rows = rows  # Adjust if needed

            # Prepare response part for this sub-query
            natural_response_prompt = (
                "You are a helpful assistant that generates a concise and direct response based solely on the SQL query result. "
                f"Result: {rows}\n" 
                "Provide a brief, straightforward answer to the query that a non-technical user can easily understand. Do not include any additional explanation."
            )

            response_generation = chat_client.chat.completions.create(
                model="gpt-4-km",
                messages=[ 
                    {"role": "system", "content": natural_response_prompt},
                ],
                max_tokens=1000  # Reduced token limit
            )

            natural_response = response_generation.choices[0].message.content.strip()
            full_response += f"{natural_response} "  # Combine the answers

        # Final natural language response
        results.append({
            "question": original_query_string,
            "response": full_response.strip()
        })

        # Retrieve user ID by email
        user_id = get_user_id_by_email(email_id)
        
        # Store the query and result in the database
        store_data_in_db(original_query_string, full_response.strip(), user_id, folder_name)
        citation_dict=[]
        # Return the combined response
        store_data_in_db(original_query_string,full_response.strip(), user_id, folder_name)
        return {"bot_answer": full_response.strip(),"citation_dict": citation_dict}

    except Error as e:
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

def extract_and_format_last_two_records(db_name: str, userEmail: str):
    try:
        # Establish the connection
        user_id = get_user_id_by_email(userEmail)
        connection = mysql.connector.connect(
            host=host_name,
            database=database_name,
            user=user_name,
            password=user_password
        )

        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            # Define the select query with filters, ordering by the latest entries
            select_query = """
            SELECT user_question, bot_answer
            FROM km_chat_history
            WHERE db_name = %s AND user_id = %s
            ORDER BY id DESC  -- Assumes `id` is a primary key or auto-increment field
            LIMIT 1
            """

            # Execute the query with parameters
            cursor.execute(select_query, (db_name, user_id))
            rows = cursor.fetchall()

            # Format the data
            formatted_data = []

            for row in rows:
                formatted_data.append({
                    "user_question": row["user_question"],
                    "bot_answer": row["bot_answer"]
                })

            return formatted_data

    except mysql.connector.Error as e:
        print(f"Error: {e}")
        return []

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
from msal import ConfidentialClientApplication
# Azure AD and Business Central API credentials


def get_access_token():
    """Authenticate and get an access token."""
    app = ConfidentialClientApplication(
        BCLIENT_ID,
        authority=BAUTHORITY,
        client_credential=BCLIENT_SECRET,
    )
    token_response = app.acquire_token_for_client(scopes=BSCOPES)
    if "access_token" in token_response:
        return token_response["access_token"]
    else:
        raise Exception(f"Failed to get access token: {token_response}")
    

def get_base_url_from_company_name(company_name):
    """Determine the base URL (Production or Sandbox) based on the company name."""
    # In this case, we are assuming "CRONUS IN" is always production for this example.
    if company_name.lower() == "cronus in":
        return f"https://api.businesscentral.dynamics.com/v2.0/{BTENANT_ID}/production/ODataV4"
    else:
        # Default to sandbox if no match is found
        return f"https://api.businesscentral.dynamics.com/v2.0/{BTENANT_ID}/sandbox/ODataV4"
# Function to create a folder for the company
def create_company_folder(company_name):
    """Create a folder with the company name if it does not exist."""
    company_folder = os.path.join(os.getcwd(), company_name)
    if not os.path.exists(company_folder):
        os.makedirs(company_folder)
    return company_folder
# Function to get user input for selecting tables or downloading all
# Function to get the list of tables (entities) from Business Central
def get_tables_from_bc(token, base_url):
    """Fetch all available tables (entities) from Business Central."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        return response.json().get("value", [])
    else:
        raise Exception(f"Failed to fetch tables: {response.status_code} - {response.text}")
    

def create_company_folder(company_name):
    """Create a folder with the company name if it does not exist."""
    company_folder = os.path.join(os.getcwd(), company_name)
    if not os.path.exists(company_folder):
        os.makedirs(company_folder)
    return company_folder
# Function to download data for each table
# Function to download data for each table
import csv
def download_data(url, access_token, output_file):
    """Download the data in its exact format."""
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        content_type = response.headers.get('Content-Type', '')
        
        if 'csv' in content_type:
            # Save as CSV file
            with open(output_file, "wb") as file:
                file.write(response.content)
            print(f"CSV file downloaded successfully: {output_file}")
        
        elif 'json' in content_type:
            # Handle JSON and convert to CSV
            data = response.json()
            if isinstance(data, dict):
                data = data.get("value", data)
            if isinstance(data, list):
                save_as_csv_from_json(data, output_file)
            else:
                print("Data is not in expected table format in JSON.")
        
        else:
            print(f"Unknown content type: {content_type}. Saving raw data.")
            with open(output_file, "wb") as file:
                file.write(response.content)
            print(f"Data saved as raw content: {output_file}")
    else:
        raise Exception(f"Failed to download data: {response.status_code} - {response.text}")

def save_as_csv_from_json(json_data, output_file):
    if not json_data:
        print("No data found to convert.")
        return
    
    # Try to determine the structure of the data
    if isinstance(json_data, list) and len(json_data) > 0:
        headers = json_data[0].keys()  # Get keys from the first element
        with open(output_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(json_data)
        print(f"CSV file saved successfully: {output_file}")
    else:
        print("Data is not in expected list format for CSV conversion.")


import os
import pandas as pd
from fastapi import HTTPException



async def respond_to_question_BC(original_query_string, folder_name, email_id):
    try:
        current_working_directory = os.getcwd()
        print(f"Current working directory: {current_working_directory}")

        # Define the desired path to store CSV/XLSX files
        db_path = os.path.join(current_working_directory, folder_name)
        print(f"Database path: {db_path}")

        # Ensure the folder is created, if not, download files
        if not os.path.exists(db_path):
            print(f"Folder {db_path} does not exist, attempting to download blobs.")
            
            # Assuming download_blobs_from_folder will download to the specified path
            result = download_blobs_from_folder_BC(CONTAINER_NAME, folder_name, CONNECTION_STRING, current_working_directory)
            
            if result:
                print("The blob is fetched successfully.")
            else:
                return {"bot_answer": "Database not created yet", "citation_dict": {}}

        # Check for folder structure issues
        files = os.listdir(db_path)
        print(f"Files in the folder: {files}")

        # Flatten the folder structure if nested folders exist
        if len(files) == 1 and os.path.isdir(os.path.join(db_path, files[0])):
            nested_folder = os.path.join(db_path, files[0])
            print(f"Flattening folder structure by moving files from {nested_folder} to {db_path}")
            for f in os.listdir(nested_folder):
                os.rename(os.path.join(nested_folder, f), os.path.join(db_path, f))
            os.rmdir(nested_folder)  # Remove the empty nested folder
            print(f"Moved files and removed the nested folder: {nested_folder}")

        # Check for CSV/XLSX files
        csv_files = [f for f in os.listdir(db_path) if f.lower().endswith((".csv", ".xls", ".xlsx"))]
        print(f"CSV/XLSX files found: {csv_files}")

        if not csv_files:
            raise HTTPException(status_code=404, detail="No CSV or Excel files found in the specified folder.")

        # Get full paths of CSV/XLSX files
        csv_paths = [os.path.join(db_path, file) for file in csv_files]

        # Initialize an empty DataFrame
        df = pd.DataFrame()

        # Process each file and merge into a single DataFrame
        for file_path in csv_paths:
            if file_path.lower().endswith(".csv"):
                # Read CSV files in chunks
                chunk_iter = pd.read_csv(file_path, chunksize=10000)
                for chunk in chunk_iter:
                    if isinstance(chunk, pd.DataFrame):
                        df = pd.concat([df, chunk], ignore_index=True)
            elif file_path.lower().endswith((".xls", ".xlsx")):
                # Read Excel files
                excel_data = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
                for sheet_name, sheet_data in excel_data.items():
                    if isinstance(sheet_data, pd.DataFrame):
                        df = pd.concat([df, sheet_data], ignore_index=True)

        # Validate the DataFrame
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found in the files to process.")

        print("DataFrame Info:")
        print(df.info())

        # Create LangChain agent with Azure OpenAI
        llm = AzureChatOpenAI(
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
            max_tokens=1000,
            temperature=0.7
        )

        # Create LangChain agent
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )

        # Run the query using the agent
        result = agent.run(original_query_string)
        print(f"Query result: {result}")
        user_id = get_user_id_by_email(email_id)
        citation_dict = []  # Add citations if needed
        final_response = {"bot_answer": result, "citation_dict": citation_dict}
        store_data_in_db(original_query_string,result, user_id, folder_name)
        return final_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")

class EmailRequest(BaseModel):
    email: str

class History(BaseModel):
    user_id: int
    database: str
 
 
# Define the input model
class Count(BaseModel):
    user_id: int
 
 
app = FastAPI()
 
 
origins = [
    "*"
]
 
# Add CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome! Middleware is working correctly."}
 
@app.post("/chat/history", response_model=List[dict])
async def get_commands(payload: History):
 
    response = extract_and_format_data(  payload.database, payload.user_id)
    return response
 
 
 
@app.post("/user/summary", response_model=dict)
async def get_commands(count: Count):
    response = count_rows_and_databases_by_user(count.user_id)
    return response
 
 

@app.post("/auth/login", response_model=dict)
async def register_user(user: UserRegistration):
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)
 
        if connection.is_connected():
            cursor = connection.cursor()
           
            # Check if the email already exists in the database and get the user_id
            email_check_query = "SELECT user_id FROM km_registration WHERE email_id = %s"
            cursor.execute(email_check_query, (user.email,))
            result = cursor.fetchone()
           
            if result:
                user_id = result[0]  # Extract the user_id from the result
                return {"message": "Email already registered", "user_id": user_id}
           
            # Generate a 10-digit random string for the password (or use user.password)
            random_string = generate_random_string(10)
           
            # Insert user details into the database
            insert_query = """
            INSERT INTO km_registration (name, email_id, password) VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (user.name, user.email, random_string))
            connection.commit()
            user_id = cursor.lastrowid  # Get the last inserted user_id
 
            return {"message": "User registered successfully", "user_id": user_id}
 
    except HTTPException as e:
        raise e
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
 
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
 
 
 

# Endpoint for user registration
@app.post("/auth/login", response_model=dict)
async def register_user(user: UserRegistration):
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)
 
        if connection.is_connected():
            cursor = connection.cursor()
                        # Generate a 10-digit random string
            random_string = generate_random_string(10)
            # Insert user details into the database
            insert_query = """
            INSERT INTO km_registration (name, email_id, password) VALUES (%s, %s , %s)
            """
            cursor.execute(insert_query, (user.name, user.email, random_string))
            connection.commit()
            user_id = cursor.lastrowid  # Get the last inserted user_id
 
            return {"message": "User registered successfully", "user_id": user_id}
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
 
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
 

 
 
@app.post("/databases")
# def list_folders(request: EmailRequest, user: dict = Depends(expiry_check)):
def list_folders(request: EmailRequest):
    try:
        email_str = request.email
        user_id = get_user_id_by_email(email_str)
        if user_id == None:
            return {"databases": []}
        print(user_id)
        db_names = get_db_names_by_user_id(user_id)
        db_names = db_names["db_names"]
        print(db_names)
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blobs = container_client.walk_blobs()
       
        # Extract folder names (prefixes)
        folders = set()
        for blob in blobs:
            folder_path = os.path.dirname(blob.name)
            if folder_path:  # Only add if it's not an empty string
                folders.add(folder_path)
        folder_list = list(folders)
        final_db_list = []
        for db in db_names:
            # if db in folder_list:
            final_db_list.append(db)
        return {"databases": final_db_list}
    except ResourceNotFoundError:
        raise HTTPException(status_code=404, detail="Container not found")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/global_databases")
# def list_folders(user: dict = Depends(expiry_check)):
def list_folders():
    try:
        db_names = get_global_db_names_by_user_id()
        db_names = db_names["db_names"]
        print(db_names)
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blobs = container_client.walk_blobs()
       
        # Extract folder names (prefixes)
        folders = set()
        for blob in blobs:
            folder_path = os.path.dirname(blob.name)
            if folder_path:  # Only add if it's not an empty string
                folders.add(folder_path)
        folder_list = list(folders)
        final_db_list = []
        for db in db_names:
            # if db in folder_list:
            final_db_list.append(db)
        return {"databases": final_db_list}
    except ResourceNotFoundError:
        raise HTTPException(status_code=404, detail="Container not found")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


import logging
@app.post("/query", response_model=QueryResponse)
# async def handle_query(request: QueryRequest, user: dict = Depends(expiry_check)):
async def handle_query(request: QueryRequest):
    try:
        logging.info(f"Database: {request.database}, Email: {request.email}")
        folder_type = get_file_type_by_database(request.database)
        logging.info(f"Folder Type: {folder_type}")

        # Select appropriate function based on folder type
        if folder_type == 'SQL':
            response_data = await respond_to_question_sql(request.query, request.database, request.email)
        elif folder_type in ['Bussiness_Central', 'csv_files']:
            response_data = await respond_to_question_BC(request.query, request.database, request.email)
        elif folder_type in ['standard_pdf', 'scanned_pdf', 'word_files']:
            response_data = await respond_to_question(request.query, request.database, request.email)
        else:
            raise HTTPException(status_code=400, detail="Unsupported folder type.")

        # Ensure response_data is in correct format for QueryResponse
        return QueryResponse(**response_data)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the query.")

BTENANT_ID = "a8801bcb-7990-408e-ab0c-e73eccd70288"
BCLIENT_ID = "328901e9-2f79-4f60-a76a-a1d033a55111"
BCLIENT_SECRET = "JiE8Q~z4PMSJQ.Dgzxm6ZZO.S-ryC0DpfSagebx4"

# Authentication parameters
BAUTHORITY = f"https://login.microsoftonline.com/{BTENANT_ID}"
BSCOPES = ["https://api.businesscentral.dynamics.com/.default"]

@app.post("/business_central/login")
async def bcentral_login(
    company_name: str = Form(...)  # Company name
):

    try:
        # Step 1: Get Access Token
        token = get_access_token()
        if not token:
            return JSONResponse(
                status_code=401,
                content={"error": "Failed to get access token"}
            )

        # Step 2: Fetch Tables
        base_url = get_base_url_from_company_name(company_name)
        tables = get_tables_from_bc(token, base_url)
        
        # Extract table names
        table_names = [table.get("name") for table in tables if "name" in table]

        # Return the response
        return {"company_name": company_name, "table_names": table_names}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Company Database is not fetched successfully"}
        )
    
@app.post("/business_central/create_database")
async def create_database(
    folder_name: str = Form(...),  # Folder name where files will be stored
    user_input: str = Form(...),
    email_id:str = Form(...),
    company_name: str = Form(...),  # Company name
    visibility: str = Form(...)

):
    
    try:
        folder_path = os.path.join(os.getcwd(), folder_name)
        selected_tables=[]
        for table_name in user_input.split(','):
            table_name = table_name.strip().lower()  # Strip and normalize case
            selected_tables.append(table_name)
        token = get_access_token()
        if not token:
            return JSONResponse(
                status_code=401,
                content={"error": "Failed to get access token"}
            )
        database_name=create_company_folder(folder_name)
        base_url = get_base_url_from_company_name(company_name)

        for table_name in selected_tables:
            table_url = f"{base_url}/{table_name}"
            print(table_url)
            token = get_access_token()
            file_name = os.path.join(database_name, f"{table_name}.csv")
            print("file_name",file_name)
            download_data(table_url,token,file_name)
            user_id = get_user_id_by_email(email_id)
            file_type='Bussiness_Central'
            
        _ = add_db_mapping(user_id, folder_name,file_type,visibility)
        print(folder_path)
        upload_files_to_blob(folder_path)

        return {
            "message": "Database Created Succussfully."
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while creating the database."}
        )
        
@app.post("/database/create")
async def upload_file(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),  # Uploaded files
    folder_name: str = Form(...),  # Folder to save files
    email: str = Form(...),  # User email
    file_type: str = Form(...),  # Type of data: 'scanned_pdf', 'standard_pdf', 'docx', etc.
    visibility: str = Form(...)
):
    try:
        # Get user_id based on the email
        user_id = get_user_id_by_email(email)
        
        # Clean up folder_name (convert to lowercase and replace spaces with underscores)
        folder_name = folder_name.lower().replace(' ', '_')
        
        # Pass the parameters to trigger_task to handle folder check and task creation
        way='upload_file'
        result = await trigger_task(background_tasks, user_id, folder_name, file_type, email, files,way,visibility)
        
        return result  # Return the response message
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def trigger_task(
    background_tasks: BackgroundTasks,
    user_id: str, folder_name: str,
    file_type: str, email: str, files: List[UploadFile],way:str,
    visibility :str

):
    # Check if the folder already exists for the user
    existing_folder = check_folder_exists(user_id, folder_name)
   
    if existing_folder:
        response = {
            "message": "Database already exists for this folder name and email."
        }
        return response
    try:
       
       
        # Create folder if it doesn't exist
        folder_path = os.path.join(os.getcwd(), folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(folder_path)
       
        # Save files to the folder
# Processing files
        for file in files:
            if way=='upload_file':
                print(f"Processing file: {file.filename}")
        # Save the file to the local system
                file_path = os.path.join(folder_path, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            elif way=='onedrive':  # If file is a string (file path)
                print(f"Processing file path: {file}")
                filename = os.path.basename(file)
                file_path = os.path.join(folder_path, filename)
                shutil.copy(file, file_path)  # Copy the file from the given path
            else:
                print(f"Unknown file type: {file}")  # This will log any non-file object
        if file_type == 'standard_pdf':
            files = os.listdir(folder_name)
            pdf_files = [f for f in files]
            total_chunks = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(folder_path, pdf_file)
                print(f"Reading {pdf_file}...")
                chunks = read_and_split_pdf(pdf_path, pdf_file)
                total_chunks += chunks  # Accumulate total chunks  
            if len(total_chunks) <= 0:
                minutes_to_wait = 0    
            minutes_to_wait = (len(total_chunks) * 2)/60  
            minutes_to_wait = round(minutes_to_wait, 2)
            # Add the background task
            background_tasks.add_task(background_task, folder_name)
            response = {
        "message": "Database created successfully",
        "eta": f"{minutes_to_wait} minutes"
        }
           
            _ = add_db_mapping(user_id, folder_name,file_type,visibility)
            return JSONResponse(content=response)
        if file_type == 'csv_files':
            files = os.listdir(folder_name)
            csv_files = [f for f in files]
            total_chunks = []
            print(csv_files)
            for csv_file in csv_files:
                csv_path = os.path.join(folder_path, csv_file)
                print(f"Reading {csv_file}...")
                print(csv_path)
                upload_files_to_blob(folder_path)
            response = {
        "message": "Database created successfully",

        }  
            _ = add_db_mapping(user_id, folder_name,file_type,visibility)
            return JSONResponse(content=response)
        if file_type == 'word_files':
            files = os.listdir(folder_name)
            word_files = [f for f in files]
            total_chunks = []
            for word_file in word_files:
                word_path = os.path.join(folder_path, word_file)
                print(f"Reading {word_file}...")
                chunks = read_and_split_word(word_path, word_file)
                total_chunks += chunks  # Accumulate total chunks  
            if len(total_chunks) <= 0:
                minutes_to_wait = 0
             
            minutes_to_wait = (len(total_chunks) * 2)/60  
            minutes_to_wait = round(minutes_to_wait, 2)
            # Add the background task
            background_tasks.add_task(background_task2, folder_name)
            response = {
    "message": "Database created successfully",
    "eta": f"{minutes_to_wait} minutes"
}
 
# Return the response as JSON
            _ = add_db_mapping(user_id, folder_name,file_type,visibility)
            return JSONResponse(content=response)
        
        if file_type == 'scanned_pdf':
            for file in files:
                
                print(file.filename)
                file_location = os.path.join(folder_path, file.filename)               
                total_chunks=process_file(file_location,file.filename,CONTAINER_NAME, CONNECTION_STRING)
                if total_chunks <= 0:
                    minutes_to_wait = 0
              
                minutes_to_wait = ((total_chunks) * 2)/60  
                minutes_to_wait = round(minutes_to_wait, 2)
            process_and_upload_embeddings(folder_path,CONTAINER_NAME, CONNECTION_STRING)
            response = {
    "message": "Database created successfully",
    "eta": f"{minutes_to_wait} minutes"

}
            _ = add_db_mapping(user_id, folder_name,file_type,visibility)
# Return the response as JSON
            return JSONResponse(content=response)
    except Exception as Argument:
        print(Argument)
        # creating/opening a file
        f = open("log.txt", "a")
        # writing in the file
        f.write(str(Argument))
        # closing the file
        f.close()
        response = {
        "message": "Database not created",
        "eta": f"0 minutes"
        }
 
        return response
 


def download_file_from_url(url):
    # Define a local folder path where the file will be saved
    local_folder = 'Downloaded_Files'
    os.makedirs(local_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Send a HEAD request to retrieve headers and determine file type
    response = requests.head(url, allow_redirects=True)

    # Initialize file extension and filename
    file_extension = ''
    filename = 'downloaded_file'

    # Check if the request was successful
    if response.status_code == 200:
        # Determine the file extension based on Content-Type
        content_type = response.headers.get('Content-Type')
        if content_type == "application/pdf":
            file_extension = 'pdf'
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_extension = 'docx'
        elif content_type == "application/octet-stream":
            file_extension = 'csv'
        elif content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            file_extension = 'xlsx'

        # Attempt to get the file name from Content-Disposition header
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition and 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[-1].strip(' "')
        else:
            print("No Content-Disposition header found or does not contain a filename.")
        
        # Add extension if not present
        if not filename.endswith(f".{file_extension}") and file_extension:
            filename = f"{filename}.{file_extension}"

        # Define the full local path for the file
        local_file_path = os.path.join(local_folder, filename)

        # Download the file content with GET request
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"File downloaded successfully as {local_file_path}")
            return local_file_path
        else:
            print(f"Failed to download the file")
            return None
    else:
        print(f"Failed to access file: {response.status_code} - {response.reason}")
        return None

@app.post("/one_drive_upload-file/")
async def upload_file(
    background_tasks: BackgroundTasks,
    file_urls: str = Form(...),  # Accept a list of file URLs
    folder_name: str = Form(...),  # Folder to save files
    email: str = Form(...),  # User email
    file_type: str = Form(...),  # Type of data: 'scanned_pdf', 'standard_pdf', 'docx', etc.
    visibility:str = Form(...)
):
    try:
        # List to store file paths of downloaded files
        file_urls=file_urls.split(',')
        downloaded_file_paths = []
        print(file_urls)
        # Download files from each URL
        for file_url in file_urls:
            
            print(f"Downloading file from URL: {file_url}")
            file_path = download_file_from_url(file_url)  # Assuming this function returns the local path
            if file_path:
                downloaded_file_paths.append(file_path)
            else:
                raise HTTPException(status_code=400, detail=f"Failed to download file from {file_url}")
        
        user_id = get_user_id_by_email(email)
        
        # Clean up folder_name (convert to lowercase and replace spaces with underscores)
        folder_name = folder_name.lower().replace(' ', '_')
        
        # Pass the parameters to trigger_task to handle folder check and task creation
        way='onedrive'
        result = await trigger_task(background_tasks, user_id, folder_name, file_type, email, downloaded_file_paths,way,visibility)
        for file_path in downloaded_file_paths:
            print(file_path)
            print("vgdhfjhjkd")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted from local storage.")
        
        return result  # Return the response message
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
#new
@app.post("/delete_database")
async def delete_database(userEmail: str, db_name: str):
    # Validate input
    if not userEmail or not db_name:
        raise HTTPException(status_code=400, detail="Invalid input: userEmail and db_name are required")

    try:
        user_id = get_user_id_by_email(userEmail)
        if not user_id:
            raise HTTPException(status_code=404, detail="User not found")

        success = delete_database_record(user_id, db_name)
        if success:
            return JSONResponse(content={"message": "Database record deleted successfully"})
        else:
            raise HTTPException(status_code=500, detail="Failed to delete database record")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
@app.post("/delete_all_database")
async def delete_database(userEmail: str):
    user_id = get_user_id_by_email(userEmail)
    if user_id:
        success = delete_all_database_record(user_id)
        if success:
            return JSONResponse(content={"message": "Database record deleted successfully"})
        else:
            raise HTTPException(status_code=500, detail="Failed to delete database record")
    else:
        raise HTTPException(status_code=404, detail="User not found")
@app.post("/fetch_id")
async def user_id_fetching(userEmail: str = Form(...)):
    print(userEmail)
    response = get_user_id_by_email(userEmail)
    if response is not None:
            return JSONResponse(content={"user_id": response})
    else:
            return JSONResponse(content={"error": "User not found"}, status_code=404)
   
 
@app.post("/sql_save_connection/")
def extract_data_from_db(email_id: str = Body(...),visibility: str = Body(...), host: str = Body(...), database: str  = Body(...), user: str  = Body(...), password: str  = Body(...)):

    file_type='SQL'

    user_id=get_user_id_by_email(email_id)
    _ = add_db_mapping(user_id, database,file_type,visibility)
    try: 
        # Connect to the database
        connection = mysql.connector.connect(**db_config)

        # Check if the connection is successful
        if connection.is_connected():
            cursor = connection.cursor()

            # Define the insert query with correct column names and backticks
            insert_query = """
            INSERT INTO km_sql (user_id, host, database_name, username, password,visibility)
VALUES (%s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    host = VALUES(host),
    database_name = VALUES(database_name),
    username = VALUES(username),
    password = VALUES(password),
    visibility = VALUES(visibility);
            """
            
            # Execute the insert query with provided parameters
            cursor.execute(insert_query, (user_id, host, database, user, password,visibility))
            connection.commit()  # Commit the transaction

            return {"status": "success", "message": "Data inserted successfully into the database."}

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        raise HTTPException(status_code=500, detail=f"Error connecting to the database: {str(e)}")

    finally:
        # Ensure the connection is closed
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")


class ContactForm(BaseModel):
    name: str
    contact_info: str
    company_name: str
    message: str
@app.post("/send-contact")
async def send_email(contact_form: ContactForm):
    try:
        # Sender email and app password from .env
        sender_email = "divyansh01agrawal@gmail.com"
        sender_password = "vgxfitwaupwrmxem"
        receiver_email = "divyansh01agrawal@gmail.com"
       
        # Construct the email
        msg = MIMEMultipart()
        msg["From"] = contact_form.contact_info
        msg["To"] = receiver_email
        msg["Subject"] = "New Contact Form Submission"
 
        body = f"""
        Name: {contact_form.name}
        Contact Info: {contact_form.contact_info}
        Company: {contact_form.company_name}
        Message: {contact_form.message}
        """
        msg.attach(MIMEText(body, "plain"))
 
        # Set up the server to send the email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)  # Using app password
            server.sendmail(sender_email, receiver_email, msg.as_string())  # Send the email
 
        return {"message": "Message sent successfully!"}
    except Exception as e:
        print(f"Error: {e}")
        return {"message": "Failed to send message"}, 500
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
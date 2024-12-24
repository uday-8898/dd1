import zipfile
import os
import re
import sys
import shutil
import itertools
import requests
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from openai import AzureOpenAI
import nltk
from nltk.tokenize import sent_tokenize
from azure.storage.blob import BlobServiceClient
import pdfplumber



client = AzureOpenAI(
  api_key = "24d6f09f8b0f44b1a1b90da2488fb417",
  api_version = "2023-05-15",
  azure_endpoint = "https://openai-production001.openai.azure.com/openai/deployments/text-embedding-3-small-km/embeddings?api-version=2023-05-15"

)


# Split Content into chunks
def chunks_string(text, tokens):
    # Initialize variables
    segments = []
    len_sum = 0
    k = 0

    # Split the text into sentences
    raw_list = sent_tokenize(text)

    # Iterate the Sentences one-by-one
    for i in range(len(raw_list)):

      # Split that sentence into tokens
      x1 = len(raw_list[i].split())

      # Cummulative length of tokens till this sentence
      len_sum = len_sum + x1

      k = k + 1

      # If no. of tokens > threshold
      if len_sum > tokens:

        ### Logic for finding how many sentences need to be repeat in current segment ###

        # Will be used for first segment only
        if i-(k+1) < 0:
            j = 0

        # Will be used for next  all segments
        else:
          j = i-(k+1)
          if len(" ".join(raw_list[j: i+1]).split()) > tokens:
            j = i-k

        # Append list of sentences to each segment
        segments.append(" ".join(raw_list[j: i]))

        # Set variables = 0
        len_sum = 0
        k = 0

      # If it is last iteration
      if i == len(raw_list)-1:
        if i-(k+1) < 0:
          j = 0

        else:
          j = i-(k+1)
          if len(" ".join(raw_list[j: i+1]).split()) > tokens:
            j = i-k

          # Append list of sentences to each segment
          segments.append(" ".join(raw_list[j: i+1]))

    return segments
def chunks_string1(text, chunk_size):
    """
    Splits the text into chunks of a specified number of words.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])
def extract_text_from_pdf(pdf_file, file_name):
    reader = PdfReader(pdf_file)
    content_chunks = []

    # Iterate through each page in the PDF
    for page_num, page in enumerate(reader.pages, start=1):
        # Extract text from the page, ensure it returns a string even if empty
        page_content = page.extract_text() or ''
        # print(f"Extracted content from page {page_num}:\n{page_content}")

        # Split content into chunks of specified word count
        chunks = chunks_string1(page_content, 200)
        # print("poiuytr",chunks)
        # print(f"Splitting page {page_num} content into chunks...")

        # Store each chunk with the page number and file name
        content_chunks.extend([
            (page_num, file_name, chunk.strip())
            for chunk in chunks if len(chunk.split()) > 2
        ])

    print("Extracted and chunked content:", content_chunks)
    return content_chunks
    


# Function to read PDF file content and split into chunks
def read_and_split_pdf(file_path, file_name, chunk_size=200):
    print(file_path)
    print(file_name)
    print("09")
    print("")
    reader = PdfReader(file_path)
    print("reader",reader.pages)
    content_chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_content = page.extract_text() or ''
        # Split content into chunks based on word count
        chunks = chunks_string(page_content, chunk_size)
        print("09")
        content_chunks.extend([(page_num,file_name, chunk.strip()) for chunk in chunks if len(chunk.split()) > 2])
        print("this is chunk",content_chunks)
    return content_chunks

# print (generate_embeddings(text_chunks[1]))
def generate_embeddings(texts, model="text-embedding-3-small-km"):
    return client.embeddings.create(input=[texts], model=model).data[0].embedding


import docx
# Set your Azure Blob Storage details
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=knowledgeminner;AccountKey=QRpbwOL0B7kMflXw3SOLrvR2CGqsxB4bKtCJ9e7QauYD/uLkqqHgRiUCjD4S6Qrwzb/OxT3q3H4M+ASt+SQirg==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "userdb"
# LOCAL_FOLDER_PATH = "folder1"  # Set your local folder path here

def extact_content_embedding_from_word_file(folder_path):
    # # Path to the folder containing PDF files
    print("234r3")
    # folder_path = '/content/Ncert'
    db_path = f"{folder_path}/{folder_path}_embedding.csv"
    print(db_path)
    old_files_list = []
    # List all files in the folder
    files = os.listdir(folder_path)
    old_df = pd.DataFrame(columns=['page_no', 'file_name', 'text',"embedding"])

    # Filter out PDF files
    word_files = [f for f in files if f.lower().endswith('.docx')]    
    print("1")

    if os.path.exists(db_path):
        old_df = pd.read_csv(f"{folder_path}/{folder_path}_embedding.csv")
        old_files_list =  old_df['file_name'].unique()
        # remove rows from old df which is no longer in folder
        for old_file in old_files_list:
            if old_file not in word_files:
                print("deleting file ", old_file)
                # Condition to remove rows where 'Column1' is greater than 3
                condition = old_df['file_name'] == old_file
                # Remove rows based on condition
                old_df = old_df[~condition]
    print("2")

    # Total number of chunks
    total_chunks = []
    embedding_list = []

    # Read each PDF file, split into chunks, and display page number and chapter name
    for word_file in word_files:
        if word_file not in old_files_list:
            word_path = os.path.join(folder_path, word_file)
            print(f"Reading {word_file}...")
            chunks = read_and_split_word(word_path, word_file)
            total_chunks += chunks  # Accumulate total chunks
            # print("Number of chunks:", len(chunks))
            # print("Chunks:")
            for page_num, file_name, chunk in total_chunks:
                print(f"Page {page_num} : Filename {file_name}: {chunk}")

    print("Total number of chunks from all PDF files:", total_chunks)
    for i, chunk in enumerate(total_chunks):

        embedding = generate_embeddings(chunk[2])
        embedding_list.append(embedding)

    # Remove empty tuples
    data = [t for t in total_chunks if t]

    # Create DataFrame
    new_df = pd.DataFrame(data, columns=['page_no', 'file_name', 'text'])
    new_df['embedding'] = embedding_list


    new_df = pd.concat([ new_df, old_df], ignore_index=True)
    new_df.to_csv(f"{folder_path}/{folder_path}_embedding.csv", index = False)

    # Upload local PDF files to Azure Blob Storage
    upload_files_to_blob(folder_path)
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(e)    
    return True

def read_and_split_word(file_path, file_name, chunk_size=200):
    # Load a Word document
    document = docx.Document(file_path)
    
    content_chunks = []
    page_num = 1  # Since python-docx does not support page-level operations, we'll use a single page number
    page_content = ""
    
    # Iterate through each paragraph in the document
    for para in document.paragraphs:
        page_content += para.text + "\n"
        print(page_content)
    
    # Split content into chunks based on word count
    chunks = chunks_string(page_content, chunk_size)
    # Add the chunks to the list if they contain more than 2 words
    content_chunks.extend([(page_num, file_name, chunk.strip()) for chunk in chunks if len(chunk.split()) > 2])
    print("hihihihihihihhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    
    return content_chunks


# Function to upload local PDF files to Azure Blob Storage, preserving folder name
def upload_files_to_blob(local_folder_path):
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    local_folder_name = os.path.basename(os.path.normpath(local_folder_path))  # Get the local folder name

    for root, _, files in os.walk(local_folder_path):
        for file in files:
            # if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, local_folder_path)  # Get relative path
                blob_name = f"{local_folder_name}/{relative_path}".replace('\\', '/')  # Include folder name in blob path
                blob_client = container_client.get_blob_client(blob_name)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True, connection_timeout=600)
                print(f"Uploaded {file} to {CONTAINER_NAME}/{blob_name}")




def extact_content_embedding_from_file(folder_path):
    # # Path to the folder containing PDF files
    # folder_path = '/content/Ncert'
    db_path = f"{folder_path}/{folder_path}_embedding.csv"
    old_files_list = []
    # List all files in the folder
    files = os.listdir(folder_path)
    old_df = pd.DataFrame(columns=['page_no', 'file_name', 'text',"embedding"])

    # Filter out PDF files
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]    


    if os.path.exists(db_path):
        old_df = pd.read_csv(f"{folder_path}/{folder_path}_embedding.csv")
        old_files_list =  old_df['file_name'].unique()
        # remove rows from old df which is no longer in folder
        for old_file in old_files_list:
            if old_file not in pdf_files:
                print("deleting file ", old_file)
                # Condition to remove rows where 'Column1' is greater than 3
                condition = old_df['file_name'] == old_file
                # Remove rows based on condition
                old_df = old_df[~condition]

    # Total number of chunks
    total_chunks = []
    embedding_list = []

    # Read each PDF file, split into chunks, and display page number and chapter name
    for pdf_file in pdf_files:
        if pdf_file not in old_files_list:
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"Reading {pdf_file}...")
            chunks = read_and_split_pdf(pdf_path, pdf_file)
            total_chunks += chunks  # Accumulate total chunks
            print("Number of chunks:", len(chunks))
            print("Chunks:")
            for page_num, file_name, chunk in total_chunks:
                print(f"Page {page_num} : Filename {file_name}: {chunk}")

    print("Total number of chunks from all PDF files:", total_chunks)
    for i, chunk in enumerate(total_chunks):

        embedding = generate_embeddings(chunk[2])
        embedding_list.append(embedding)

    # Remove empty tuples
    data = [t for t in total_chunks if t]

    # Create DataFrame
    new_df = pd.DataFrame(data, columns=['page_no', 'file_name', 'text'])
    new_df['embedding'] = embedding_list


    new_df = pd.concat([ new_df, old_df], ignore_index=True)
    new_df.to_csv(f"{folder_path}/{folder_path}_embedding.csv", index = False)

    # Upload local PDF files to Azure Blob Storage
    upload_files_to_blob(folder_path)
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(e)    
    return True


def extact_content_embedding_from_csv_file(folder_path):
    # # Path to the folder containing PDF files
    # folder_path = '/content/Ncert'
    db_path = f"{folder_path}/{folder_path}_embedding.csv"
    old_files_list = []
    # List all files in the folder
    files = os.listdir(folder_path)
    print(files)
    old_df = pd.DataFrame(columns=['page_no', 'file_name', 'text',"embedding"])

    # Filter out PDF files
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]    


    if os.path.exists(db_path):
        old_df = pd.read_csv(f"{folder_path}/{folder_path}_embedding.csv")
        old_files_list =  old_df['file_name'].unique()
        # remove rows from old df which is no longer in folder
        for old_file in old_files_list:
            if old_file not in pdf_files:
                print("deleting file ", old_file)
                # Condition to remove rows where 'Column1' is greater than 3
                condition = old_df['file_name'] == old_file
                # Remove rows based on condition
                old_df = old_df[~condition]

    # Total number of chunks
    total_chunks = []
    embedding_list = []

    # Read each PDF file, split into chunks, and display page number and chapter name
    for pdf_file in pdf_files:
        if pdf_file not in old_files_list:
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"Reading {pdf_file}...") 
            chunks = extract_text_from_pdf(pdf_path, pdf_file)
            total_chunks += chunks  # Accumulate total chunks
            print("Number of chunks:", len(chunks))
            print("Chunks:")
            for page_num, file_name, chunk in total_chunks:
                print(f"Page {page_num} : Filename {file_name}: {chunk}")

    print("Total number of chunks from all PDF files:", total_chunks)
    for i, chunk in enumerate(total_chunks):

        embedding = generate_embeddings(chunk[2])
        embedding_list.append(embedding)

    # Remove empty tuples
    data = [t for t in total_chunks if t]

    # Create DataFrame
    new_df = pd.DataFrame(data, columns=['page_no', 'file_name', 'text'])
    new_df['embedding'] = embedding_list


    new_df = pd.concat([ new_df, old_df], ignore_index=True)
    new_df.to_csv(f"{folder_path}/{folder_path}_embedding.csv", index = False)

    # Upload local PDF files to Azure Blob Storage
    upload_files_to_blob(folder_path)
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(e)    
    return True
from reportlab.lib.pagesizes import letter

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
def convert_excel_to_pdf(excel_file, folder_path):
    print("Starting conversion...")
    
    # Extract the base name of the file without the extension
    base_name = os.path.splitext(os.path.basename(excel_file))[0]
    
    # Define the output PDF file path
    pdf_file = os.path.join(folder_path, f"{base_name}.pdf")

    # Determine file type and load data accordingly
    if excel_file.lower().endswith('.csv'):
        df = pd.read_csv(excel_file)
    elif excel_file.lower().endswith(('.xls', '.xlsx')):
        try:
            df = pd.read_excel(excel_file)
        except ImportError:
            raise ImportError("Missing optional dependency 'openpyxl'. Use 'pip install openpyxl' to install it.")
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    # Create a PDF canvas
    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter
    print("PDF canvas created.")
    
    # Set initial Y-position for the text
    y_position = height - 50  # Start near the top of the page
    
    # Write each row of data onto the PDF
    for index, row in df.iterrows():
        row_string = ', '.join(str(item) for item in row)
        c.drawString(30, y_position, row_string)
        y_position -= 15  # Move down for the next row
        
        # Start a new page if the Y-position goes below the margin
        if y_position < 50:
            c.showPage()  # Create a new page
            y_position = height - 50  # Reset Y-position for the new page

    # Save the PDF
    c.save()
    print("PDF saved successfully.")
    return pdf_file

# extact_content_embedding_from_file(r'C:\Users\RoshanKumar\OneDrive - Meridian Solutions\Desktop\ChatbotAPI\Ncert')
# df = pd.read_csv("embedding.csv")
# print(df.file_name.unique())
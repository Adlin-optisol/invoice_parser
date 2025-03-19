import os
import tempfile
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from dotenv import load_dotenv
from elsai_core.model import AzureOpenAIConnector
from elsai_core.config.loggerConfig import setup_logger

# Initialize logger
logger = setup_logger()

# Set page config
st.set_page_config(
    page_title="Invoice Parser",
    page_icon="📄",
    layout="wide"
)


def extract_content_from_pdf(pdf_path, endpoint, key):
    """
    Extract tables and text from a PDF file using Azure Document Intelligence.
    
    Args:
        pdf_path (str): Path to the PDF file
        endpoint (str): Azure Document Intelligence endpoint
        key (str): Azure Document Intelligence API key
        
    Returns:
        tuple: (extracted_text, extracted_tables)
    """
    logger.info(f"Starting extraction from PDF: {os.path.basename(pdf_path)}")
    
    try:
        if not endpoint or not key:
            logger.error("Azure Document Intelligence credentials not found")
            raise ValueError("Azure Document Intelligence credentials not found")
        
        # Initialize the Document Intelligence client
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )
        logger.debug("Document Intelligence client initialized")
        
        # Process the PDF file
        with open(pdf_path, "rb") as f:
            logger.info("Beginning document analysis")
            poller = document_intelligence_client.begin_analyze_document("prebuilt-layout", body=f)
        
        # Get the result
        logger.info("Waiting for document analysis to complete")
        result = poller.result()
        logger.info("Document analysis completed successfully")
        
        # Extract text content
        logger.debug("Extracting text content")
        extracted_text = extract_text(result)
        
        # Extract tables
        logger.debug("Extracting tables")
        extracted_tables = extract_tables(result)
        
        logger.info(f"Extraction complete. Found {len(extracted_text)} pages of text and {len(extracted_tables)} tables")
        return extracted_text, extracted_tables
    
    except Exception as e:
        logger.error(f"Error extracting content from PDF: {str(e)}", exc_info=True)
        raise

def extract_text(result):
    """
    Extract text content from the analysis result.
    
    Args:
        result: The result from Document Intelligence analysis
        
    Returns:
        dict: Dictionary containing text content by page
    """
    logger.debug("Starting text extraction from analysis result")
    text_content = {}
    
    # Extract text from paragraphs (most reliable for formatted text)
    if result.paragraphs:
        logger.debug(f"Found {len(result.paragraphs)} paragraphs to extract")
        # Sort paragraphs by their position in the document
        sorted_paragraphs = sorted(
            result.paragraphs, 
            key=lambda p: (p.spans[0].offset if p.spans else 0)
        )
        
        for paragraph in sorted_paragraphs:
            page_numbers = [region.page_number for region in paragraph.bounding_regions] if paragraph.bounding_regions else []
            
            for page_num in page_numbers:
                if page_num not in text_content:
                    text_content[page_num] = []
                
                text_content[page_num].append({
                    "type": "paragraph",
                    "content": paragraph.content,
                    "role": paragraph.role if hasattr(paragraph, "role") else None
                })
    
    # If no paragraphs, extract text from pages
    if not text_content and result.pages:
        logger.debug(f"No paragraphs found, extracting from {len(result.pages)} pages")
        for page in result.pages:
            page_num = page.page_number
            text_content[page_num] = []
            
            if page.lines:
                for line in page.lines:
                    text_content[page_num].append({
                        "type": "line",
                        "content": line.content
                    })
    
    logger.debug(f"Text extraction complete. Extracted text from {len(text_content)} pages")
    return text_content

def extract_tables(result):
    """
    Extract tables from the analysis result.
    
    Args:
        result: The result from Document Intelligence analysis
        
    Returns:
        list: List of dictionaries containing table data
    """
    logger.debug("Starting table extraction from analysis result")
    extracted_tables = []
    
    if result.tables:
        logger.debug(f"Found {len(result.tables)} tables to extract")
        for table_idx, table in enumerate(result.tables):
            logger.debug(f"Processing table {table_idx+1} with {table.row_count} rows and {table.column_count} columns")
            # Create a table representation
            table_data = {
                "table_id": table_idx,
                "row_count": table.row_count,
                "column_count": table.column_count,
                "page_numbers": [],
                "cells": []
            }
            
            # Add page numbers where this table appears
            if table.bounding_regions:
                for region in table.bounding_regions:
                    if region.page_number not in table_data["page_numbers"]:
                        table_data["page_numbers"].append(region.page_number)
            
            # Extract cell data
            for cell in table.cells:
                cell_data = {
                    "row_index": cell.row_index,
                    "column_index": cell.column_index,
                    "content": cell.content,
                    "is_header": cell.kind == "columnHeader" if hasattr(cell, "kind") else False,
                    "spans": cell.column_span if hasattr(cell, "column_span") else 1
                }
                table_data["cells"].append(cell_data)
            
            extracted_tables.append(table_data)
            logger.debug(f"Extracted table {table_idx+1} with {len(table_data['cells'])} cells")
    
    logger.debug(f"Table extraction complete. Extracted {len(extracted_tables)} tables")
    return extracted_tables

def process_pdf(file_path, vision_endpoint, vision_key):
    """
    Process a PDF file.
    
    Args:
        file_path: Path to the PDF file
        vision_endpoint: Azure Document Intelligence endpoint
        vision_key: Azure Document Intelligence API key
        
    Returns:
        str: LLM processed results
    """
    file_name = os.path.basename(file_path)
    logger.info(f"Processing PDF file: {file_name}")
    
    try:
        # Extract content from PDF
        logger.info("Extracting content from PDF")
        text_content, tables = extract_content_from_pdf(file_path, vision_endpoint, vision_key)
        
        # Process with LLM
        logger.info("Initializing LLM connector")
        connector = AzureOpenAIConnector()
        llm = connector.connect_azure_open_ai(deploymentname="gpt-4o-mini")
        logger.info("LLM connector initialized")
        
        # Prepare text content for LLM
        text_content_str = ""
        for page_num in sorted(text_content.keys()):
            text_content_str += f"\n\n### Page {page_num}\n"
            for item in text_content[page_num]:
                text_content_str += f"{item['content']}\n"
        
        # Prepare tables for LLM
        tables_str = ""
        if tables:
            tables_str += "\n\n## Tables\n"
            for i, table in enumerate(tables):
                tables_str += f"\n### Table {i+1}\n"
                tables_str += f"Pages: {', '.join(map(str, table['page_numbers']))}\n\n"
                
                # Create simple text representation of table
                rows = table["row_count"]
                cols = table["column_count"]
                grid = [["" for _ in range(cols)] for _ in range(rows)]
                
                for cell in table["cells"]:
                    row = cell["row_index"]
                    col = cell["column_index"]
                    grid[row][col] = cell["content"]
                
                for row in grid:
                    tables_str += " | ".join(row) + "\n"
        
        # Combine content for LLM
        
        logger.info("Generating prompt for LLM processing")
        prompt = f"""
            Extract all key information from this invoice and organize it into well-structured markdown tables.

            The content is as follows:
                        Text from the document : {text_content_str}
                        Tables from the document : {tables_str}

            Based on the information provided above, create multiple markdown tables that capture all relevant invoice details. Include:

            1. Create separate tables for different categories of information (invoice details, vendor information, customer information, line items, payment details, etc.)
            2. Name each table appropriately based on its content
            3. Include ALL information present in the invoice
            4. Group related information logically

            Requirements:
            - Use proper markdown table syntax with headers, separators, and appropriate alignment
            - Include clear table titles above each table
            - Replace any empty/missing values with "N/A" rather than leaving cells blank
            - Ensure field names accurately describe the data they contain
            - Format numbers, dates, and currency values consistently
            - Create as many tables as needed to properly organize the information
            - Do not include any explanatory text outside the tables themselves

            Example format:

            ### Invoice Details
            | Field | Value |
            |:------|:------|
            | Invoice Number | INV-12345 |
            | Date | 2023-05-15 |

            ### Line Items
            | Item Description | Quantity | Unit Price | Total |
            |:-----------------|:--------:|:----------:|:------|
            | Product A | 2 | $50.00 | $100.00 |
            | Service B | 5 | $75.00 | $375.00 |
            """
        
        logger.info("Sending request to LLM")
        response = llm.invoke(prompt)
        result = response.content
        logger.info(f"Received response from LLM ({len(result)} characters)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing PDF {file_name}: {str(e)}", exc_info=True)
        return f"Error processing PDF: {str(e)}"

def main():
    """
    Main Streamlit application
    """
    st.title("Invoice Parser")
    st.markdown("Upload invoice PDFs to extract structured data")
    
    # Check for environment variables
    vision_endpoint = st.secrets["VISION_ENDPOINT"]
    vision_key = st.secrets["VISION_KEY"]
    
    if not vision_endpoint or not vision_key:
        st.error("Azure Document Intelligence credentials not found in environment variables. Please set VISION_ENDPOINT and VISION_KEY.")
        return
    
    # File uploader section
    st.subheader("Upload PDF Invoices")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s)")
        
        # Process files when user clicks the button
        if st.button("Process Files"):
            for uploaded_file in uploaded_files:
                # Create progress bar for this file
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"Processing: {uploaded_file.name}...")
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    progress_bar.progress(25)
                    status_text.text(f"Analyzing document: {uploaded_file.name}")
                    
                    # Process the PDF
                    result = process_pdf(temp_file_path, vision_endpoint, vision_key)
                    
                    progress_bar.progress(100)
                    status_text.text(f"Completed: {uploaded_file.name}")
                    
                    # Display results in an expander
                    with st.expander(f"Results for {uploaded_file.name}", expanded=True):
                        st.markdown(result)
                        
                        # Add download button for the results
                        st.download_button(
                            label="Download results as markdown",
                            data=result,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_results.md",
                            mime="text/markdown"
                        )
                    
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.text(f"Error processing: {uploaded_file.name}")
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                
            st.success("All files processed!")
    else:
        st.info("Please upload PDF invoice files to begin")
    
if __name__ == "__main__":
    try:
        logger.info("Streamlit application starting")
        main()
        logger.info("Streamlit application session ended")
    except Exception as e:
        logger.critical(f"Unhandled exception in Streamlit application: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
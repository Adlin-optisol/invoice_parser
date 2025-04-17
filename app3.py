import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from elsai_core.model import AzureOpenAIConnector
from elsai_core.extractors.azure_document_intelligence import AzureDocumentIntelligence
from elsai_core.config.loggerConfig import setup_logger
from elsai_core.prompts import PezzoPromptRenderer
load_dotenv()
# Initialize logger
logger = setup_logger()

# Set page config
st.set_page_config(
    page_title="Invoice Parser",
    page_icon="📄",
    layout="wide"
)

def extract_content_from_pdf(pdf_path):
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

        doc_processor = AzureDocumentIntelligence(pdf_path)
        logger.debug("Extracting text")
        extracted_text = doc_processor.extract_text()
        
        # Extract tables
        logger.debug("Extracting tables")
        extracted_tables = doc_processor.extract_tables()
        
        logger.info(f"Extraction complete. Found {len(extracted_text)} pages of text and {len(extracted_tables)} tables")
        return extracted_text, extracted_tables
    
    except Exception as e:
        logger.error(f"Error extracting content from PDF: {str(e)}", exc_info=True)
        raise
def format_table(tables):
    """
    Format the table into a string representation.
    
    Args:
        table: The table object to format
    Returns:
        str: Formatted table string
    """
    tables_str = ""
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
    return tables_str
def process_pdf(file_path):
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
        text_content, tables = extract_content_from_pdf(file_path)
        tables_str = format_table(tables)
        logger.info("Content extraction completed")
        # Process with LLM
        logger.info("Initializing LLM connector")
        connector = AzureOpenAIConnector()
        llm = connector.connect_azure_open_ai(deploymentname="gpt-4o-mini")
        logger.info("LLM connector initialized")
           
        # Combine content for LLM
        logger.info("Generating prompt for LLM processing")
        renderer = PezzoPromptRenderer(
            api_key=st.secrets["PEZZO_API_KEY"],
            project_id=st.secrets["PEZZO_PROJECT_ID"],
            environment=st.secrets["PEZZO_ENVIRONMENT"],
            server_url=st.secrets["PEZZO_SERVER_URL"]
        )
        prompt = renderer.get_prompt("InvoiceParsingPrompt") 
        prompt_txt = prompt+ f"""The content is as follows: Text from the document : {text_content} , Tables from the document : {tables_str}"""
        print(prompt_txt)
        logger.info("Sending request to LLM")
        response = llm.invoke(prompt_txt)
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
                    result = process_pdf(temp_file_path).replace("```markdown","").replace("```","")
                    
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

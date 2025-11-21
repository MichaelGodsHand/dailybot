from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import sys
import os
from io import BytesIO
from pathlib import Path
import uuid

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Please install it using:")
    print("  pip install pymupdf")
    sys.exit(1)

try:
    from PIL import Image
    import pytesseract
except ImportError:
    print("OCR dependencies not found. Please install them using:")
    print("  pip install pillow pytesseract")
    sys.exit(1)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not found. Please install it using:")
    print("  pip install chromadb")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not found. Please install it using:")
    print("  pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not found. Please install it using:")
    print("  pip install python-dotenv")
    sys.exit(1)

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("boto3 not found. Please install it using:")
    print("  pip install boto3")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("requests not found. Please install it using:")
    print("  pip install requests")
    sys.exit(1)

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
except ImportError:
    print("reportlab not found. Please install it using:")
    print("  pip install reportlab")
    sys.exit(1)

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    print("twilio not found. Please install it using:")
    print("  pip install twilio")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Configure Tesseract path for Windows
TESSERACT_FOUND = False
if sys.platform == 'win32':
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        TESSERACT_FOUND = True
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                TESSERACT_FOUND = True
                break
else:
    TESSERACT_FOUND = True

# Initialize FastAPI
app = FastAPI(
    title="PDF Text Extraction & Query API with WhatsApp",
    description="Extract text from PDFs, store in ChromaDB, query with GPT-4o, and send via WhatsApp",
    version="1.0.0"
)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="pdf_documents",
    metadata={"hnsw:space": "cosine"}
)

# Initialize OpenAI client (with error handling)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("\n⚠️  WARNING: OPENAI_API_KEY not found in .env file or environment!")
    print("The /query endpoint will not work without an API key.")
    openai_client = None
else:
    openai_client = OpenAI(api_key=openai_api_key)

# Initialize S3 client (with error handling)
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

if not aws_access_key or not aws_secret_key:
    print("\n⚠️  WARNING: AWS credentials not found in .env file or environment!")
    print("The S3 upload functionality will not work without AWS credentials.")
    s3_client = None
else:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "real-estate-brochures-tenori")

# Initialize Twilio client (with error handling)
twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

if not twilio_account_sid or not twilio_auth_token:
    print("\n⚠️  WARNING: Twilio credentials not found in .env file or environment!")
    print("The WhatsApp sending functionality will not work without Twilio credentials.")
    twilio_client = None
else:
    twilio_client = TwilioClient(twilio_account_sid, twilio_auth_token)

# Pydantic model for query request
class QueryRequest(BaseModel):
    query: str
    number: str


def send_whatsapp_message(to_number: str, summary: str, pdf_url: str = None):
    """
    Send WhatsApp message with summary and optional PDF attachment.
    
    Args:
        to_number: Recipient's phone number (e.g., "918438232949")
        summary: Text summary to send
        pdf_url: Optional URL of PDF to attach
        
    Returns:
        dict: Status of message sending
    """
    if twilio_client is None:
        print(f"  ⚠ Twilio client not configured, skipping WhatsApp message")
        return {"status": "skipped", "reason": "Twilio not configured"}
    
    try:
        print(f"\n  📱 Sending WhatsApp message...")
        print(f"    → To: {to_number}")
        print(f"    → Summary length: {len(summary)} chars")
        print(f"    → PDF URL: {pdf_url if pdf_url else 'None'}")
        print(f"    → PDF attached: {'Yes' if pdf_url else 'No'}")
        
        # Format phone number for WhatsApp
        formatted_number = f"whatsapp:{to_number}" if not to_number.startswith('whatsapp:') else to_number
        
        # Prepare message body - just the summary, no prefix
        message_body = summary
        
        # Create message parameters
        message_params = {
            'from_': twilio_whatsapp_number,
            'body': message_body,
            'to': formatted_number
        }
        
        # Add PDF if available - IMPORTANT: Twilio needs publicly accessible URLs
        if pdf_url:
            # Test if URL is accessible
            print(f"    → Testing PDF URL accessibility...")
            try:
                test_response = requests.head(pdf_url, timeout=5)
                print(f"    → PDF URL status code: {test_response.status_code}")
                if test_response.status_code == 200:
                    message_params['media_url'] = [pdf_url]
                    print(f"    → PDF URL is accessible, adding to message")
                else:
                    print(f"    ⚠ PDF URL returned {test_response.status_code}, may not be accessible to Twilio")
                    message_params['media_url'] = [pdf_url]  # Try anyway
            except Exception as url_test_error:
                print(f"    ⚠ Could not test URL accessibility: {url_test_error}")
                message_params['media_url'] = [pdf_url]  # Try anyway
        
        # Send message
        print(f"    → Sending message via Twilio...")
        message = twilio_client.messages.create(**message_params)
        
        print(f"  ✓ WhatsApp message sent successfully!")
        print(f"    → Message SID: {message.sid}")
        print(f"    → Status: {message.status}")
        print(f"    → Direction: {message.direction}")
        
        # Check for any errors
        if hasattr(message, 'error_code') and message.error_code:
            print(f"    ⚠ Error code: {message.error_code}")
            print(f"    ⚠ Error message: {message.error_message}")
        
        return {
            "status": "success",
            "message_sid": message.sid,
            "twilio_status": message.status,
            "pdf_url_sent": pdf_url
        }
        
    except Exception as e:
        print(f"  ✗ Failed to send WhatsApp message: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }


def extract_text_with_ocr(page, page_num, pdf_name):
    """Extract text from a PDF page using OCR."""
    global TESSERACT_FOUND
    
    if not TESSERACT_FOUND:
        print(f"    ⚠ OCR skipped (Tesseract not available)")
        return ""
    
    try:
        print(f"    🔍 Running OCR on {pdf_name}&{page_num}...", end="", flush=True)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        ocr_text = pytesseract.image_to_string(img, lang='eng')
        print(" ✓ Done")
        return ocr_text.strip()
    except pytesseract.TesseractNotFoundError:
        TESSERACT_FOUND = False
        print(" ✗ Failed (Tesseract not found)")
        return ""
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return ""


def extract_text_from_pdf(pdf_bytes, pdf_filename, use_ocr=True):
    """Extract text from PDF file page by page."""
    pdf_text = {}
    page_objects = {}  # Store page objects for S3 upload
    
    try:
        pdf_name = Path(pdf_filename).stem
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        print(f"\n{'='*80}")
        print(f"📄 Processing PDF: {pdf_filename}")
        print(f"📊 Total pages: {len(doc)}")
        print(f"🔧 OCR enabled: {use_ocr}")
        print(f"{'='*80}\n")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_identifier = f"{pdf_name}&{page_num + 1}"
            
            print(f"  📖 Page {page_num + 1}/{len(doc)} ({page_identifier})")
            
            print(f"    📝 Extracting regular text...", end="", flush=True)
            regular_text = page.get_text()
            regular_char_count = len(regular_text.strip())
            print(f" ✓ ({regular_char_count} chars)")
            
            ocr_text = ""
            ocr_char_count = 0
            if use_ocr:
                ocr_text = extract_text_with_ocr(page, page_num + 1, pdf_name)
                ocr_char_count = len(ocr_text.strip())
                if ocr_text:
                    print(f"      ✓ OCR extracted {ocr_char_count} chars")
            
            if regular_text.strip() and ocr_text.strip():
                if regular_text.strip() in ocr_text or len(regular_text.strip()) < 50:
                    combined_text = ocr_text
                    print(f"    💡 Using OCR text only (regular text contained in OCR)")
                else:
                    combined_text = f"{regular_text}\n\n[OCR Text from Images:]\n{ocr_text}"
                    print(f"    💡 Combined regular + OCR text")
            elif ocr_text.strip():
                combined_text = f"[OCR Text:]\n{ocr_text}"
                print(f"    💡 Using OCR text only")
            else:
                combined_text = regular_text
                print(f"    💡 Using regular text only")
            
            pdf_text[page_identifier] = combined_text
            page_objects[page_identifier] = page
            
            total_chars = len(combined_text.strip())
            print(f"    ✅ Page {page_identifier} complete - Total: {total_chars} chars")
            print(f"    {'-'*76}")
        
        print(f"\n{'='*80}")
        print(f"✅ PDF Processing Complete: {pdf_filename}")
        print(f"📊 Total pages processed: {len(pdf_text)}")
        print(f"{'='*80}\n")
        
        return pdf_text, page_objects, doc
        
    except Exception as e:
        print(f"\n❌ Error extracting text from {pdf_filename}: {str(e)}\n")
        raise Exception(f"Error extracting text from {pdf_filename}: {str(e)}")


def store_in_chromadb(page_identifier, text, pdf_name, page_number):
    """Store extracted text in ChromaDB with metadata. Prevents duplicates."""
    try:
        print(f"    💾 Storing {page_identifier} in ChromaDB...", end="", flush=True)
        
        # Check if this page already exists (prevent duplicates)
        existing = collection.get(
            where={"page_identifier": page_identifier},
            limit=1
        )
        
        if existing['ids']:
            # Page exists, update it instead of creating duplicate
            print(f" (updating existing)...", end="", flush=True)
            collection.update(
                ids=[existing['ids'][0]],
                documents=[text],
                metadatas=[{
                    "page_identifier": page_identifier,
                    "pdf_name": pdf_name,
                    "page_number": str(page_number),
                    "char_count": str(len(text))
                }]
            )
            print(" ✓ Updated")
        else:
            # New page, add it
            doc_id = f"{page_identifier}"
            collection.add(
                documents=[text],
                metadatas=[{
                    "page_identifier": page_identifier,
                    "pdf_name": pdf_name,
                    "page_number": str(page_number),
                    "char_count": str(len(text))
                }],
                ids=[doc_id]
            )
            print(" ✓ Stored")
        
        return True
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return False


def upload_page_to_s3(page, page_identifier):
    """
    Convert PDF page to image and upload to S3.
    
    Args:
        page: PyMuPDF page object
        page_identifier: Page identifier (e.g., "E-Brochure-1&3")
    
    Returns:
        str: S3 URL of uploaded image, or None if failed
    """
    if s3_client is None:
        print(f"    ⚠ S3 client not configured, skipping upload for {page_identifier}")
        return None
    
    try:
        print(f"    📤 Uploading {page_identifier} to S3...", end="", flush=True)
        
        # Render page as high-quality image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        # S3 key (filename in bucket)
        s3_key = f"{page_identifier}.png"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=img_bytes,
            ContentType='image/png'
        )
        
        # Generate S3 URL
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        
        print(f" ✓ Uploaded")
        return s3_url
        
    except ClientError as e:
        print(f" ✗ Failed (S3 Error: {e.response['Error']['Message']})")
        return None
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return None


def create_compiled_pdf_from_images(s3_image_urls, user_number, query):
    """
    Download images from S3, compile them into a single PDF, and upload to S3.
    
    Args:
        s3_image_urls: List of S3 image URLs
        user_number: User's phone number
        query: The original query
        
    Returns:
        str: S3 URL of compiled PDF, or None if failed
    """
    if s3_client is None:
        print(f"  ⚠ S3 client not configured, cannot create compiled PDF")
        return None
    
    try:
        print(f"  📄 Creating compiled PDF from {len(s3_image_urls)} images...")
        
        if not s3_image_urls:
            return None
        
        # Create a temporary PDF file
        import tempfile
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()
        
        # Create PDF with ReportLab
        c = canvas.Canvas(temp_pdf_path, pagesize=A4)
        page_width, page_height = A4
        
        # Download and add each image to PDF
        for idx, img_url in enumerate(s3_image_urls, 1):
            print(f"    📥 Processing image {idx}/{len(s3_image_urls)}...", end="", flush=True)
            
            try:
                # Download image from S3
                response = requests.get(img_url, timeout=30)
                response.raise_for_status()
                
                # Open image with PIL
                img = Image.open(BytesIO(response.content))
                
                # Calculate dimensions to fit page while maintaining aspect ratio
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                
                # Fit to page with margins
                margin = 20
                available_width = page_width - (2 * margin)
                available_height = page_height - (2 * margin)
                
                if available_width * aspect <= available_height:
                    # Width is limiting factor
                    display_width = available_width
                    display_height = available_width * aspect
                else:
                    # Height is limiting factor
                    display_height = available_height
                    display_width = available_height / aspect
                
                # Center image on page
                x = (page_width - display_width) / 2
                y = (page_height - display_height) / 2
                
                # Draw image
                img_reader = ImageReader(BytesIO(response.content))
                c.drawImage(img_reader, x, y, width=display_width, height=display_height)
                
                # Add new page if not last image
                if idx < len(s3_image_urls):
                    c.showPage()
                
                print(" ✓")
                
            except Exception as e:
                print(f" ✗ Failed: {str(e)}")
                continue
        
        # Save PDF
        c.save()
        print(f"  ✓ PDF created successfully")
        
        # Upload to S3
        print(f"  📤 Uploading compiled PDF to S3...", end="", flush=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"query_{user_number}_{timestamp}.pdf"
        
        # Read PDF file
        with open(temp_pdf_path, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
        
        # Upload to S3 (using the same bucket as page images)
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"compiled_pdfs/{pdf_filename}",  # Store in a subfolder for organization
            Body=pdf_bytes,
            ContentType='application/pdf',
            Metadata={
                'user_number': user_number,
                'query': query[:200],  # Truncate long queries
                'page_count': str(len(s3_image_urls))
            }
        )
        
        # Generate presigned URL (valid for 1 hour) for Twilio to access
        try:
            compiled_pdf_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': S3_BUCKET_NAME,
                    'Key': f"compiled_pdfs/{pdf_filename}"
                },
                ExpiresIn=3600  # URL valid for 1 hour
            )
            print(f" ✓ Uploaded")
            print(f"  ✓ Compiled PDF URL (presigned): {compiled_pdf_url[:100]}...")
        except Exception as presign_error:
            # Fallback to regular URL if presigning fails
            compiled_pdf_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/compiled_pdfs/{pdf_filename}"
            print(f" ✓ Uploaded")
            print(f"  ⚠ Could not generate presigned URL, using regular URL: {compiled_pdf_url}")
        
        # Clean up temp file
        os.unlink(temp_pdf_path)
        
        return compiled_pdf_url
        
    except Exception as e:
        print(f"  ✗ Failed to create compiled PDF: {str(e)}")
        return None


@app.post("/extract")
async def extract_text(
    files: List[UploadFile] = File(...),
    use_ocr: bool = True
):
    """
    Extract text from multiple PDF files and store in ChromaDB.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    print("\n" + "="*80)
    print(f"🚀 Starting batch PDF extraction")
    print(f"📁 Total files received: {len(files)}")
    print(f"🔧 OCR enabled: {use_ocr}")
    print(f"🔧 OCR available: {TESSERACT_FOUND}")
    print("="*80)
    
    results = {}
    errors = []
    stored_count = 0
    s3_upload_count = 0
    
    for idx, file in enumerate(files, 1):
        print(f"\n📦 Processing file {idx}/{len(files)}: {file.filename}")
        
        if not file.filename.lower().endswith('.pdf'):
            error_msg = f"{file.filename}: Not a PDF file"
            errors.append(error_msg)
            print(f"  ❌ {error_msg}")
            continue
        
        try:
            print(f"  📥 Reading file contents...")
            contents = await file.read()
            print(f"  ✓ File size: {len(contents)} bytes")
    
    # Extract text
            extracted_text, page_objects, doc = extract_text_from_pdf(contents, file.filename, use_ocr)
            
            # Store each page in ChromaDB and upload to S3
            pdf_name = Path(file.filename).stem
            print(f"\n  💾 Storing pages in ChromaDB and uploading to S3...")
            
            for page_identifier, text in extracted_text.items():
                # Extract page number from identifier
                page_number = int(page_identifier.split('&')[1])
                
                # Store in ChromaDB
                if store_in_chromadb(page_identifier, text, pdf_name, page_number):
                    stored_count += 1
                
                # Upload page image to S3
                page = page_objects[page_identifier]
                s3_url = upload_page_to_s3(page, page_identifier)
                if s3_url:
                    s3_upload_count += 1
            
            # Close the document
            doc.close()
            
            results.update(extracted_text)
            print(f"  ✅ Successfully extracted and stored {len(extracted_text)} pages from {file.filename}")
            
        except Exception as e:
            error_msg = f"{file.filename}: {str(e)}"
            errors.append(error_msg)
            print(f"  ❌ Failed: {error_msg}")
    
    response = {
        "status": "success" if results else "failed",
        "total_files_processed": len(files),
        "total_pages_extracted": len(results),
        "total_pages_stored_in_db": stored_count,
        "total_pages_uploaded_to_s3": s3_upload_count,
        "ocr_enabled": use_ocr,
        "ocr_available": TESSERACT_FOUND,
        "page_identifiers": list(results.keys())
    }
    
    if errors:
        response["errors"] = errors
    
    print("\n" + "="*80)
    print(f"🎉 Batch extraction complete!")
    print(f"✅ Success: {len(results)} pages extracted")
    print(f"💾 Stored: {stored_count} pages in ChromaDB")
    print(f"📤 Uploaded: {s3_upload_count} pages to S3")
    if errors:
        print(f"⚠ Errors: {len(errors)} files failed")
    print("="*80 + "\n")
    
    return JSONResponse(content=response)


@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query the stored PDF documents using GPT-4o, ChromaDB, and send results via WhatsApp.
    """
    query = request.query
    user_number = request.number
    
    print("\n" + "="*80)
    print(f"🔍 Processing query: {query}")
    print(f"📱 User number: {user_number}")
    print("="*80)
    
    try:
        # Query ChromaDB for relevant documents - increase results to ensure we get good matches
        print(f"  📊 Searching ChromaDB for relevant documents...")
        
        # Extract PDF name from query to filter results to relevant PDF only
        query_lower = query.lower()
        pdf_name_filter = None
        
        # Common property name patterns and their PDF identifiers
        # Map common property name variations to their PDF file names
        # Order matters: longer/more specific patterns should be checked first
        pdf_name_mappings = [
            ('palm premiere', 'palm-premiere-brochure'),
            ('palmpremiere', 'palm-premiere-brochure'),
            ('palmpremier', 'palm-premiere-brochure'),
        ]
        
        # Try to detect PDF name from query (check for property names, longest first)
        for property_name, pdf_name in pdf_name_mappings:
            if property_name in query_lower:
                pdf_name_filter = pdf_name
                print(f"  🔍 Detected property: '{property_name}' → Filtering to PDF: {pdf_name}")
                break
        
        # If no specific property detected, check all stored PDFs to infer
        if not pdf_name_filter:
            # Get all unique PDF names from collection
            all_results = collection.get(limit=1000)
            unique_pdf_names = set()
            if all_results.get('metadatas'):
                for meta in all_results['metadatas']:
                    if meta.get('pdf_name'):
                        unique_pdf_names.add(meta['pdf_name'])
            
            # Try fuzzy match on query text
            for stored_pdf in unique_pdf_names:
                pdf_lower = stored_pdf.lower().replace('-', ' ').replace('_', ' ')
                # Check if any significant words from PDF name appear in query
                pdf_words = pdf_lower.split()
                query_words = query_lower.split()
                if any(word in query_words for word in pdf_words if len(word) > 4):
                    pdf_name_filter = stored_pdf
                    print(f"  🔍 Matched query to PDF: {pdf_name_filter}")
                    break
        
        # Enhanced search query - add keywords that might help find relevant content
        enhanced_query = query
        if 'kitchen' in query_lower:
            enhanced_query = f"{query} kitchen specifications details features"
        elif 'bedroom' in query_lower:
            enhanced_query = f"{query} bedroom specifications details features"
        elif 'bathroom' in query_lower:
            enhanced_query = f"{query} bathroom specifications details features"
        
        # Build query with optional PDF filter
        query_params = {
            'query_texts': [enhanced_query],
            'n_results': 20  # Get more results to ensure we find relevant pages
        }
        
        # Add PDF filter if detected
        if pdf_name_filter:
            query_params['where'] = {'pdf_name': pdf_name_filter}
            print(f"  ✓ Filtering to PDF: {pdf_name_filter}")
        else:
            print(f"  ⚠ No specific PDF detected, searching all PDFs")
        
        results = collection.query(**query_params)
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            print(f"  ⚠ No documents found in ChromaDB")
            return JSONResponse(content={
                "status": "no_results",
                "summary": "No relevant documents found in the database.",
                "pages": [],
                "s3_images": [],
                "compiled_pdf_url": None,
                "whatsapp_status": {"status": "skipped", "reason": "No results"}
            })
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # Deduplicate by page_identifier (keep best relevance score)
        seen_pages = {}
        deduplicated_docs = []
        deduplicated_metas = []
        deduplicated_distances = []
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            page_id = meta['page_identifier']
            relevance = 1 - dist
            
            # Keep only the best relevance score for each page
            if page_id not in seen_pages or relevance > seen_pages[page_id]['relevance']:
                seen_pages[page_id] = {
                    'doc': doc,
                    'meta': meta,
                    'dist': dist,
                    'relevance': relevance
                }
        
        # Sort by relevance (highest first) and take top results
        sorted_pages = sorted(seen_pages.items(), key=lambda x: x[1]['relevance'], reverse=True)
        
        # Take top 15 unique pages
        for page_id, page_data in sorted_pages[:15]:
            deduplicated_docs.append(page_data['doc'])
            deduplicated_metas.append(page_data['meta'])
            deduplicated_distances.append(page_data['dist'])
        
        print(f"  ✓ Found {len(deduplicated_docs)} unique relevant documents (from {len(documents)} total results)")
        
        # Prepare context for GPT-4o
        context_parts = []
        pages_used = []
        
        for i, (doc, metadata, distance) in enumerate(zip(deduplicated_docs, deduplicated_metas, deduplicated_distances)):
            page_identifier = metadata['page_identifier']
            pages_used.append(page_identifier)
            context_parts.append(f"[Source: {page_identifier}]\n{doc}\n")
            relevance_score = 1 - distance
            print(f"    📄 {page_identifier} (relevance: {relevance_score:.3f})")
        
        context = "\n".join(context_parts)
        
        # Check if OpenAI client is available
        if openai_client is None:
            print(f"  ❌ OpenAI client not configured. Cannot process query.")
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."
            )
        
        # Query GPT-4o with structured output
        print(f"\n  🤖 Querying GPT-4o...")
        
        system_prompt = """You are a helpful assistant that answers questions based on PDF document content.
You MUST use the provided document excerpts to answer the user's query. 
Your response MUST be in JSON format with exactly two fields:
1. "summary": A comprehensive answer based ONLY on the provided document excerpts
2. "pages_used": An array of page identifiers (e.g., ["palm-premiere-brochure&19", "palm-premiere-brochure&14"]) that you used from the provided excerpts

CRITICAL RULES:
- You MUST include ALL page identifiers that contain information relevant to answering the query
- Even if information appears in multiple pages, include all those page identifiers
- If the document excerpts contain relevant information, you MUST use it and list the source pages
- Only exclude pages that have NO relevant information at all
- Be thorough - if kitchen details are in page 19, you MUST include "palm-premiere-brochure&19" in pages_used"""

        user_prompt = f"""Query: "{query}"

Below are document excerpts from a real estate brochure. Answer the query using ONLY information from these excerpts. Be comprehensive and include all relevant details.

Document excerpts:
{context}

REQUIREMENTS:
1. Extract ALL relevant information from the excerpts to answer: "{query}"
2. List ALL page identifiers that contain information relevant to your answer
3. Be thorough - include pages even if they only have partial information

Respond in JSON format:
{{
  "summary": "Your detailed answer based on the excerpts above. Include specific details, measurements, features mentioned.",
  "pages_used": ["page-identifier-1", "page-identifier-2", ...]
}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent, factual responses
            max_tokens=2000,  # Increased for more detailed answers
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        import json
        result = json.loads(response.choices[0].message.content)
        summary = result.get("summary", "")
        pages_actually_used = result.get("pages_used", [])
        
        # Fallback: If GPT returns no pages but we have context, use pages with high relevance
        if not pages_actually_used and pages_used:
            print(f"  ⚠ GPT returned 0 pages, using pages with relevance > 0.5 as fallback")
            # Use pages with relevance score > 0.5
            for i, (metadata, distance) in enumerate(zip(deduplicated_metas, deduplicated_distances)):
                relevance = 1 - distance
                if relevance > 0.5:
                    pages_actually_used.append(metadata['page_identifier'])
        
        # Ensure pages_actually_used are valid (exist in pages_used)
        pages_actually_used = [p for p in pages_actually_used if p in pages_used]
        
        # If still empty, use top 5 most relevant pages
        if not pages_actually_used:
            print(f"  ⚠ Still no pages, using top 5 most relevant pages as fallback")
            pages_actually_used = pages_used[:5]
        
        print(f"  ✓ GPT-4o response generated")
        print(f"  📄 GPT-4o used {len(pages_actually_used)} out of {len(pages_used)} retrieved pages")
        
        # Log which pages were actually used
        for page in pages_actually_used:
            print(f"    ✓ Used: {page}")
        
        # Fetch S3 URLs for the pages used
        s3_images = []
        if s3_client is not None:
            print(f"\n  🔗 Fetching S3 image URLs...")
            for page_identifier in pages_actually_used:
                s3_key = f"{page_identifier}.png"
                s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
                
                # Verify if the object exists in S3 (optional but recommended)
                try:
                    s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                    s3_images.append(s3_url)
                    print(f"    ✓ Found: {s3_url}")
                except ClientError:
                    print(f"    ⚠ Not found in S3: {page_identifier}")
        else:
            print(f"\n  ⚠ S3 client not configured, skipping S3 URL fetching")
        
        # Create compiled PDF from images
        compiled_pdf_url = None
        if s3_client is not None and s3_images:
            print(f"\n  📚 Creating compiled PDF...")
            compiled_pdf_url = create_compiled_pdf_from_images(s3_images, user_number, query)
        else:
            print(f"\n  ⚠ Skipping compiled PDF creation (S3 not configured or no images)")
        
        # Send WhatsApp message
        whatsapp_status = send_whatsapp_message(user_number, summary, compiled_pdf_url)
        
        print(f"\n{'='*80}")
        print(f"✅ Query completed successfully")
        print(f"📄 Pages retrieved: {len(pages_used)}, Pages used: {len(pages_actually_used)}")
        print(f"🖼️  S3 images found: {len(s3_images)}")
        print(f"📚 Compiled PDF: {'Created' if compiled_pdf_url else 'Failed'}")
        print(f"📱 WhatsApp: {whatsapp_status['status']}")
        print(f"{'='*80}\n")
        
        return JSONResponse(content={
            "status": "success",
            "summary": summary,
            "pages": pages_actually_used,
            "s3_images": s3_images,
            "compiled_pdf_url": compiled_pdf_url,
            "whatsapp_status": whatsapp_status
        })
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Check for required API keys and credentials
    print("\n" + "="*80)
    print("🔧 Configuration Check")
    print("="*80)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not found!")
    else:
        print("✓ OpenAI API key configured")
    
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("⚠️  WARNING: AWS credentials not found!")
    else:
        print("✓ AWS credentials configured")
    
    if not os.getenv("TWILIO_ACCOUNT_SID") or not os.getenv("TWILIO_AUTH_TOKEN"):
        print("⚠️  WARNING: Twilio credentials not found!")
    else:
        print("✓ Twilio credentials configured")
    
    print("\nPlease ensure your .env file contains:")
    print("  OPENAI_API_KEY=your-openai-api-key")
    print("  AWS_ACCESS_KEY_ID=your-aws-access-key")
    print("  AWS_SECRET_ACCESS_KEY=your-aws-secret-key")
    print("  AWS_REGION=us-east-1")
    print("  S3_BUCKET_NAME=your-bucket-name")
    print("  TWILIO_ACCOUNT_SID=your-twilio-account-sid")
    print("  TWILIO_AUTH_TOKEN=your-twilio-auth-token")
    print("  TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Upload to blob storage and support Form Recognizer (no embeddings)

import os
import glob
import html
import io
import re
import time
from pypdf import PdfReader, PdfWriter
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents.indexes.models import SemanticSettings
from azure.search.documents.indexes.models import PrioritizedFields
from dotenv import load_dotenv

# Set env vars
load_dotenv()
deployment=os.environ['AZURE_OPENAI_DEPLOYMENT']

# Chunking settings
MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100

storage_creds = os.environ['AZURE_STORAGE_KEY']
container_name = os.environ['AZURE_STORAGE_CONTAINER']
storage_account=os.environ['AZURE_STORAGE_ACCOUNT']
form_recognizer_name=os.environ['AZURE_FORM_RECOGNIZER_NAME']
form_recognizer_key=os.environ['AZURE_FORM_RECOGNIZER_KEY']
use_form_recognizer = os.environ['USE_FORM_RECOGNIZER'].lower() == "true"
search_service_name = os.environ['AZURE_SEARCH_SERVICE_NAME']
search_creds = AzureKeyCredential(os.environ['AZURE_SEARCH_KEY'])
index_name = os.environ['AZURE_SEARCH_INDEX_NAME']

def blob_name_from_file_page(filename, page = 0):
    if os.path.splitext(filename)[1].lower() == ".pdf":
        return os.path.splitext(os.path.basename(filename))[0] + f"-{page}" + ".pdf"
    else:
        return os.path.basename(filename)
    
def upload_blobs(filename):
    blob_service = BlobServiceClient(account_url=f"https://{storage_account}.blob.core.windows.net", credential=storage_creds)
    blob_container = blob_service.get_container_client(container_name)
    if not blob_container.exists():
        blob_container.create_container()

    # if file is PDF split into pages and upload each page as a separate blob
    if os.path.splitext(filename)[1].lower() == ".pdf":
        reader = PdfReader(filename)
        pages = reader.pages
        for i in range(len(pages)):
            blob_name = blob_name_from_file_page(filename, i)
            print(f"\tUploading blob for page {i} -> {blob_name}")
            f = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(pages[i])
            writer.write(f)
            f.seek(0)
            blob_container.upload_blob(blob_name, f, overwrite=True)
    else:
        blob_name = blob_name_from_file_page(filename)
        with open(filename,"rb") as data:
            blob_container.upload_blob(blob_name, data, overwrite=True)

def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html

def get_document_text(filename):
    offset = 0
    page_map = []
    if not use_form_recognizer:
        if os.path.splitext(filename)[1].lower() == ".pdf":
            reader = PdfReader(filename)
            pages = reader.pages
            for page_num, p in enumerate(pages):
                page_text = p.extract_text()
                page_map.append((page_num, offset, page_text))
                offset += len(page_text)
        else:
            with open(filename, "r") as f:
                page_text = f.read()
                page_map.append((0, offset, page_text))                
    else:
        print(f"Extracting text from '{filename}' using Azure Form Recognizer")
        form_recognizer_client = DocumentAnalysisClient(
            endpoint=f"https://{form_recognizer_name}.cognitiveservices.azure.com/", 
            credential=form_recognizer_key, 
            headers={"x-ms-useragent": "azure-search-chat-demo/1.0.0"})
        
        with open(filename, "rb") as f:
            poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", document = f)
        form_recognizer_results = poller.result()

        for page_num, page in enumerate(form_recognizer_results.pages):
            tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1]*page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >=0 and idx < page_length:
                            table_chars[idx] = table_id

            # build page text by replacing charcters in table spans with table html
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += form_recognizer_results.content[page_offset + idx]
                elif not table_id in added_tables:
                    page_text += table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)

            page_text += " "
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)

    return page_map

def split_text(filename, page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
    print(f"Splitting '{filename}' into sections")

    def find_page(offset):
        l = len(page_map)
        for i in range(l - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return l - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table")
        if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
            # If the section ends with an unclosed table, we need to start the next section with the table.
            # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
            # If last table starts inside SECTION_OVERLAP, keep overlapping
            print(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP
        
    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))

def create_sections(filename, page_map):
    for i, (section, pagenum) in enumerate(split_text(filename,page_map)):
        yield {
            "id": re.sub("[^0-9a-zA-Z_-]","_",f"{filename}-{i}"),
            "content": section,
            "category": "SEARCH",
            "sourcepage": blob_name_from_file_page(filename, pagenum),
            "sourcefile": filename
        }

def create_search_index():
    print(f"Ensuring search index {index_name} exists")
    index_client = SearchIndexClient(endpoint=f"https://{search_service_name}.search.windows.net/",
                                     credential=search_creds)
    if index_name not in index_client.list_index_names():
        search_index = SearchIndex(
            name=index_name,
            fields=[
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String", analyzer_name="en.microsoft"),
                SimpleField(name="category", type="Edm.String", filterable=True, facetable=True),
                SimpleField(name="sourcepage", type="Edm.String", filterable=True, facetable=True),
                SimpleField(name="sourcefile", type="Edm.String", filterable=True, facetable=True)
            ],
            semantic_settings=SemanticSettings(
                configurations=[SemanticConfiguration(
                    name='default',
                    prioritized_fields=PrioritizedFields(
                        title_field=None, prioritized_content_fields=[SemanticField(field_name='content')]))])
        )
        print(f"Creating {index_name} search index")
        index_client.create_index(search_index)
    else:
        print(f"Search index {index_name} already exists")
        
def index_sections(filename, sections):
    print(f"Indexing sections from '{filename}' into search index '{index_name}'")
    search_client = SearchClient(endpoint=f"https://{search_service_name}.search.windows.net/",
                                    index_name=index_name,
                                    credential=search_creds)
    i = 0
    batch = []
    for s in sections:
        batch.append(s)
        i += 1
        if i % 1000 == 0:
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            batch = []

    if len(batch) > 0:
        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
        
def remove_from_index(filename):
    print(f"Removing sections from '{filename or '<all>'}' from search index '{index}'")
    search_client = SearchClient(endpoint=f"https://{search_service_name}.search.windows.net/",
                                    index_name=index_name,
                                    credential=search_creds)
    while True:
        filter = None if filename == None else f"sourcefile eq '{os.path.basename(filename)}'"
        r = search_client.search("", filter=filter, top=1000, include_total_count=True)
        if r.get_count() == 0:
            break
        r = search_client.delete_documents(documents=[{ "id": d["id"] } for d in r])
        print(f"\tRemoved {len(r)} sections from index")
        # It can take a few seconds for search results to reflect changes, so wait a bit
        time.sleep(2)


def main():
    try:
        # Ensures search index exists
        create_search_index()    
        #Uploads blobs and indexes them
        for filename in glob.glob("data/*.*"):
            print(f"Processing '{filename}'")
            upload_blobs(filename)
            page_map = get_document_text(filename)
            sections = create_sections(filename, page_map)
            index_sections(filename, sections)
        print ("Upload completed")
    except Exception as ex:
        print ("Upload failed")
        print(ex)

if(__name__ == "__main__"):    
    main()

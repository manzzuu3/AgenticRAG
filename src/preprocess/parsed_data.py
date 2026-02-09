# parsed_data.py
import pdfplumber
import pandas as pd
from typing import List, Tuple
from pydantic import BaseModel
import os
import json


HEADER_HEIGHT_RATIO = 0.08
FOOTER_HEIGHT_RATIO = 0.08


class PageElement(BaseModel):

    element_id: str
    page_number: int
    element_type: str  
    markdown: str
    raw_text: str

class PageMarkdown(BaseModel):

    page_number: int
    markdown: str


def df_to_markdown(df: pd.DataFrame) -> str:

    return df.to_markdown(index=False)

def crop_header_and_footer(page):

    top_y = page.height * HEADER_HEIGHT_RATIO
    bottom_y = page.height * (1 - FOOTER_HEIGHT_RATIO)
    
    bbox = (0, top_y, page.width, bottom_y)
    
    return page.crop(bbox)

def process_pdf(pdf_path: str) -> Tuple[List[PageMarkdown], List[PageElement]]:

    pages_markdown = []
    all_elements = []

    print(f"Opening PDF: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"   Found {total_pages} pages. Extracting content...")
        
        for i, page in enumerate(pdf.pages):
            page_number = i + 1
            
            if page_number > 1:
                page = crop_header_and_footer(page)
            
            page_content_buffer = []  
            element_count = 0
            
            raw_text = (page.extract_text() or "").strip()
            
            if raw_text:
                element_id = f"page_{page_number}_seq_{element_count}"
                element_count += 1
                
                # Store text element
                text_element = PageElement(
                    element_id=element_id,
                    page_number=page_number,
                    element_type="text",
                    markdown=raw_text,
                    raw_text=raw_text
                )
                all_elements.append(text_element)
                

                page_content_buffer.append(f"## Page {page_number}\n\n{raw_text}\n")


            tables = page.extract_tables(table_settings={
                "horizontal_strategy": "lines", 
                "vertical_strategy": "lines"
            })
            
            for table_idx, table_data in enumerate(tables):
                if not table_data:
                    continue  


                headers = table_data[0]
                rows = table_data[1:]
                df = pd.DataFrame(rows, columns=headers)
                
                table_md = df_to_markdown(df)
                
                element_id = f"page_{page_number}"
                element_count += 1
                caption = f"Table {table_idx + 1} on Page {page_number}"

                table_element = PageElement(
                    element_id=element_id,
                    page_number=page_number,
                    element_type="table",
                    markdown=f"{caption}\n\n{table_md}",
                    raw_text="\n".join(["\t".join(map(str, row)) for row in table_data])
                )
                all_elements.append(table_element)
                
                page_content_buffer.append(f"\n### {caption}\n\n{table_md}\n")

            if page_content_buffer:
                full_page_text = "\n".join(page_content_buffer).strip()
                page_obj = PageMarkdown(page_number=page_number, markdown=full_page_text)
                pages_markdown.append(page_obj)

    return pages_markdown, all_elements


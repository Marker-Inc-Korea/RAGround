import asyncio
import base64
import itertools
import json
import os
from typing import List, Optional, Tuple

import aiohttp
import fitz  # PyMuPDF

from autorag.data.parse.base import parser_node
from autorag.utils.util import process_batch


@parser_node
def clova_ocr(
	data_path_list: List[str],
	url: Optional[str] = None,
	api_key: Optional[str] = None,
	batch: int = 8,
	table_detection: bool = False,
) -> Tuple[List[str], List[str]]:
	url = os.getenv("CLOVA_URL", None) if url is None else url
	if url is None:
		raise KeyError(
			"Please set the URL for Clova OCR in the environment variable CLOVA_URL "
			"or directly set it on the config YAML file."
		)

	api_key = os.getenv("CLOVA_API_KEY", None) if api_key is None else api_key
	if api_key is None:
		raise KeyError(
			"Please set the API key for Clova OCR in the environment variable CLOVA_API_KEY "
			"or directly set it on the config YAML file."
		)

	image_datas, image_names = zip(
		*[pdf_to_images(data_path) for data_path in data_path_list]
	)
	image_data_list = list(itertools.chain(*image_datas))
	image_name_list = list(itertools.chain(*image_names))

	tasks = [
		clova_ocr_pure(image_data, image_name, url, api_key, table_detection)
		for image_data, image_name in zip(image_data_list, image_name_list)
	]
	loop = asyncio.get_event_loop()
	results = loop.run_until_complete(process_batch(tasks, batch))

	texts, names = zip(*results)
	return list(texts), list(names)


async def clova_ocr_pure(
	image_data: bytes,
	image_name: str,
	url: str,
	api_key: str,
	table_detection: bool = False,
) -> Tuple[str, str]:
	session = aiohttp.ClientSession()
	headers = {"X-OCR-SECRET": api_key, "Content-Type": "application/json"}

	# Convert image data to base64
	image_base64 = base64.b64encode(image_data).decode("utf-8")

	# Set data
	data = {
		"version": "V2",
		"requestId": "sample_id",
		"timestamp": 0,
		"images": [{"format": "png", "name": "sample_image", "data": image_base64}],
		"enableTableDetection": table_detection,
	}

	async with session.post(url, headers=headers, data=json.dumps(data)) as response:
		resp_json = await response.json()
		if "images" not in resp_json:
			raise RuntimeError(
				f"Invalid response from Clova API: {resp_json['detail']}"
			)

		table_html = json_to_html_table(resp_json["images"][0]["tables"][0]["cells"])
		page_text = extract_text_from_fields(resp_json["images"][0]["fields"])

		if table_html:
			page_text += f"\n\ntable html:\n{table_html}"

		await session.close()
		return page_text, image_name


def pdf_to_images(pdf_path) -> Tuple[List[bytes], List[str]]:
	pdf_document = fitz.open(pdf_path)
	pdf_name = pdf_path.split("/")[-1]

	image_data_list, image_name_list = [], []
	for page_num in range(len(pdf_document)):
		page = pdf_document.load_page(page_num)

		# get image data
		pix = page.get_pixmap()
		img_data = pix.tobytes("png")
		image_data_list.append(img_data)

		img_name = f"{pdf_name}_{page_num + 1}.png"
		image_name_list.append(img_name)

	return image_data_list, image_name_list


def extract_text_from_fields(fields):
	text = ""
	for field in fields:
		text += field["inferText"]
		if field["lineBreak"]:
			text += "\n"
		else:
			text += " "
	return text.strip()


def json_to_html_table(json_data):
	# Initialize the HTML table
	html = '<table border="1">\n'
	# Determine the number of rows and columns
	max_row = max(cell["rowIndex"] + cell["rowSpan"] for cell in json_data)
	max_col = max(cell["columnIndex"] + cell["columnSpan"] for cell in json_data)
	# Create a 2D array to keep track of merged cells
	table = [["" for _ in range(max_col)] for _ in range(max_row)]
	# Fill the table with cell data
	for cell in json_data:
		row = cell["rowIndex"]
		col = cell["columnIndex"]
		row_span = cell["rowSpan"]
		col_span = cell["columnSpan"]
		cell_text = (
			" ".join(
				line["inferText"] for line in cell["cellTextLines"][0]["cellWords"]
			)
			if cell["cellTextLines"]
			else ""
		)
		# Place the cell in the table
		table[row][col] = {"text": cell_text, "rowSpan": row_span, "colSpan": col_span}
		# Mark merged cells as occupied
		for r in range(row, row + row_span):
			for c in range(col, col + col_span):
				if r != row or c != col:
					table[r][c] = None
	# Generate HTML from the table array
	for row in table:
		html += "  <tr>\n"
		for cell in row:
			if cell is None:
				continue
			if cell == "":
				html += "    <td></td>\n"
			else:
				row_span_attr = (
					f' rowspan="{cell["rowSpan"]}"' if cell["rowSpan"] > 1 else ""
				)
				col_span_attr = (
					f' colspan="{cell["colSpan"]}"' if cell["colSpan"] > 1 else ""
				)
				html += f'    <td{row_span_attr}{col_span_attr}>{cell["text"]}</td>\n'
		html += "  </tr>\n"
	html += "</table>"
	return html

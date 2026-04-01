import time
from dataclasses import dataclass, field
from enum import Enum

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrAutoOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode


class OCREngine(str, Enum):
    EASY_OCR = "easy_ocr"
    NONE = "none"


@dataclass
class OCROptions:
    engine: OCREngine = field(default=OCREngine.EASY_OCR)
    do_ocr: bool = False
    do_table_structure: bool = False
    do_cell_matching: bool = False
    table_structure_mode: TableFormerMode = TableFormerMode.ACCURATE
    generate_picture_images: bool = False
    generate_page_images: bool = False
    lang: list[str] | None = None
    # EasyOCR specific options
    force_full_page_ocr: bool = False

    def to_docling_pipeline_options(self) -> dict:
        return {
            "do_ocr": self.do_ocr,
            "do_table_structure": self.do_table_structure,
            "table_structure_options": TableStructureOptions(
                do_cell_matching=self.do_cell_matching, mode=self.table_structure_mode
            ),
            "generate_picture_images": self.generate_picture_images,
            "generate_page_images": self.generate_page_images,
            "ocr_options": EasyOcrOptions(lang=self.lang or ["auto"], force_full_page_ocr=self.force_full_page_ocr)
            if self.engine == OCREngine.EASY_OCR
            else OcrAutoOptions(),
        }


@dataclass
class OCRAcceleratorOptions:
    num_threads: int = 4
    device: AcceleratorDevice = AcceleratorDevice.AUTO

    def to_accelerator_options(self) -> AcceleratorOptions:
        return AcceleratorOptions(num_threads=self.num_threads, device=self.device)


class DoclingParser:
    """
    A parser that uses the Docling library to convert PDF documents to Markdown format.

    Notes:
        How to speed it up: https://github.com/docling-project/docling/discussions/245#discussioncomment-11156560

    """

    def __init__(self, ocr_options: OCROptions, ocr_acceleration_options: OCRAcceleratorOptions) -> None:
        pipeline_kwargs = ocr_options.to_docling_pipeline_options()
        pipeline_options = PdfPipelineOptions(**pipeline_kwargs)

        # Enable hardware acceleration if available
        acceleration_options = ocr_acceleration_options.to_accelerator_options()
        pipeline_options.accelerator_options = acceleration_options

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend)
            }
        )

    def parse_to_markdown(
        self,
        file_paths: list[str],
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER,
        traverse_pictures: bool = False,
    ) -> dict[str, str]:
        """
        Parses the given file paths to Markdown format.

        Args:
            file_paths: List of file paths to parse.
            image_mode: How to handle images in the markdown output.
            traverse_pictures: Whether to include picture images in the markdown output (if generated).

        Notes:
            ImageRefMode.PLACEHOLDER: Shows image placeholders with descriptions (if generated).
            ImageRefMode.EMBEDDED: Embeds base64-encoded images in the markdown.
            ImageRefMode.REFERENCED: Saves images to files and references them.
            https://github.com/docling-project/docling/issues/1878
            https://github.com/docling-project/docling/issues/1654

        Performance Note:
            Per-page export (export_to_markdown(page_no=X)) has significant overhead (~26x slower
            than exporting all pages at once). Most time (99%+) is spent in document conversion,
            not export. Parallelizing per-page export actually makes things worse.
        """
        file_to_markdown = {}
        for file_path in file_paths:
            result = self.converter.convert(source=file_path)
            file_to_markdown[file_path] = result.document.export_to_markdown(
                image_mode=image_mode, traverse_pictures=traverse_pictures
            )
        return file_to_markdown


if __name__ == "__main__":
    file_paths = ["2501.17887v1.pdf"]
    documents = list(file_paths)

    ocr_options = OCROptions(
        engine=OCREngine.EASY_OCR,
        do_ocr=False,
        do_table_structure=True,
        do_cell_matching=False,
        table_structure_mode=TableFormerMode.FAST,
        generate_picture_images=False,
        generate_page_images=False,
        lang=["en"],
        force_full_page_ocr=False,
    )

    ocr_acceleration_options = OCRAcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)

    parser = DoclingParser(ocr_options, ocr_acceleration_options)

    start_time = time.time()
    file_to_markdown = parser.parse_to_markdown(file_paths=file_paths)
    took = time.time() - start_time
    print(f"--- Document converted in {took:.2f} seconds ---")

    for file_path, markdown in file_to_markdown.items():
        print(f"--- Markdown for {file_path} ---")
        print(markdown)

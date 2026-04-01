# PDF Parsing

> For PDF inputs, OpenAI says the system can extract text directly from the PDF, and on vision-capable models it can also
> extract page images. In ChatGPT specifically, Enterprise supports PDF “Visual Retrieval,” so it can use both text and
> embedded visuals; for other ChatGPT plans, OpenAI says document handling is generally text-based retrieval, which means
> digital text is extracted and images are discarded. Direct text extraction is much faster than running full OCR over every page.

- https://www.linkedin.com/feed/update/urn:li:activity:7421507563424432128/
- [Hands-on Retrieval Augmented Generation - 1st Part: Document Indexing](https://theembedding1.substack.com/p/hands-on-retrieval-augmented-generation)
- [Docling vs Unstructured.io: Document Parsing for Enterprise AI Teams](https://www.ertas.ai/blog/docling-vs-unstructured-io-enterprise)

## Docling:

- https://docling-project.github.io/docling/examples/
- [Docling: Great quality, but painfully slow](https://www.reddit.com/r/LocalLLaMA/comments/1mpmhxj/docling_great_quality_but_painfully_slow/)
  - https://github.com/docling-project/docling/discussions/245#discussioncomment-11156560
- Used by data-pizza:
  - https://github.com/datapizza-labs/datapizza-ai/blob/0c5a090d6954aa3518e9227c465f99ba1427ae9d/datapizza-ai-modules/parsers/docling/datapizza/modules/parsers/docling/docling_parser.py#L21
  - https://github.com/datapizza-labs/datapizza-ai/blob/0c5a090d6954aa3518e9227c465f99ba1427ae9d/docs/API%20Reference/Modules/Parsers/docling_parser.md?plain=1#L8
  - https://github.com/datapizza-labs/datapizza-ai/blob/0c5a090d6954aa3518e9227c465f99ba1427ae9d/datapizza-ai-modules/parsers/docling/datapizza/modules/parsers/docling/ocr_options.py#L63

### Granite Docling

Docling uses a multi-stage pipeline with specialized models, layout, tables, equations, image descriptions, plus a PDF parser/OCR (like EasyOCR).
With Granite Docling, we’re working on an ultra-compact VLM that can handle all of these tasks in a single model (no parsing needed).
The VLM approach has the advantage of capturing the full document context, while the multi-stage pipeline offers more modularity and flexibility.
Other VLMs out there export to formats that we think aren't complete, so with SmolDocling and now Granite Docling we predict in a format that can be transformed to DoclingDocument.
Docling actually supports both: you can run inference with the traditional multi-stage setup or with Granite Docling, depending on what works best for your use case.

- https://www.ibm.com/granite/docs/models/docling
- https://huggingface.co/collections/ibm-granite/granite-docling
- [Why is granite-docling-258M so slow?](https://huggingface.co/ibm-granite/granite-docling-258M/discussions/37)
- https://medium.com/data-science-in-your-pocket/ibm-granite-docling-the-best-ai-document-parser-0f4e3de079cb

## Alternatives to Docling and Unstructured.io:
  - https://github.com/bytedance/Dolphin
  - https://github.com/Zipstack/unstract (LLMWhisperer) - https://unstract.com/blog/docling-alternative/
  - https://github.com/microsoft/markitdown
  - https://github.com/kreuzberg-dev/kreuzberg - They claim it's faster than Docling
  - https://github.com/opendataloader-project/opendataloader-pdf - They claim it converts PDFs into Markdown at 100+ pages/sec on CPU
  - https://huggingface.co/zai-org/GLM-OCR 0.9B (It seems bullshit)

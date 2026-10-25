# PDF page provenance

PDF ingestion retains each extracted page as an explicit boundary. Newly ingested
PDF chunks are built inside one page and carry a content-minimizing locator:

```json
{
  "page_number": 7,
  "page_start_utf8": 184,
  "page_end_utf8": 462,
  "page_text_sha256": "..."
}
```

The offsets are half-open byte positions into the normalized text returned by
`pypdf` for that page. The stored chunk is the exact byte slice at that range, so a
reviewer with the same extracted page can reproduce it. Retrieval returns the
locator, and `evidence-snapshot/v2` commits a canonical digest of it alongside the
chunk ID, source digest, exact text digest, and byte length. Changing the page,
range, or page-text digest therefore changes the authenticated generation-admission
evidence.

## Migration and compatibility

Opening an existing SQLite database adds nullable locator columns without rewriting
old chunks. Existing non-PDF chunks and previously indexed PDF chunks remain
queryable with `source_locator: null`. Reingest a PDF into a fresh index generation
to obtain page-native locators; migration does not invent provenance that was never
recorded.

## Trust boundary

The page digest identifies normalized extracted text; it does not authenticate the
PDF publisher or prove that extraction preserved visual reading order. Scanned PDFs
still require an OCR pipeline, and table/layout coordinates are outside this
increment. For high-assurance review, retain the original PDF by content digest and
pin the parser version used to produce page text.

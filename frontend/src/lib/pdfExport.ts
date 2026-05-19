const A4_WIDTH_PT = 595.28;
const A4_HEIGHT_PT = 841.89;
const PAGE_MARGIN_PT = 24;
const EXPORT_SCALE = 2;

type PdfImage = {
  bytes: Uint8Array;
  width: number;
  height: number;
  displayHeightPt: number;
};

type SaveFilePickerOptions = {
  suggestedName?: string;
  types?: Array<{
    description: string;
    accept: Record<string, string[]>;
  }>;
};

type WritableFileStream = {
  write: (data: Blob) => Promise<void>;
  close: () => Promise<void>;
};

type FileSystemFileHandle = {
  createWritable: () => Promise<WritableFileStream>;
};

type SavePickerWindow = Window & {
  showSaveFilePicker?: (options?: SaveFilePickerOptions) => Promise<FileSystemFileHandle>;
};

function makeCanvas(width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.ceil(width));
  canvas.height = Math.max(1, Math.ceil(height));
  return canvas;
}

function fillWhite(context: CanvasRenderingContext2D, width: number, height: number) {
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
}

function canvasToJpegBytes(canvas: HTMLCanvasElement): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          reject(new Error("Unable to encode the score image."));
          return;
        }
        resolve(new Uint8Array(await blob.arrayBuffer()));
      },
      "image/jpeg",
      0.92
    );
  });
}

function collectScoreCanvas(host: HTMLElement): HTMLCanvasElement {
  const renderedCanvases = Array.from(host.querySelectorAll("canvas")).filter(
    (canvas) => canvas.width > 0 && canvas.height > 0
  );

  if (!renderedCanvases.length) {
    throw new Error("The score is not ready to export yet.");
  }

  const hostRect = host.getBoundingClientRect();
  const canvasRects = renderedCanvases.map((canvas) => {
    const rect = canvas.getBoundingClientRect();
    return {
      canvas,
      left: rect.left - hostRect.left,
      top: rect.top - hostRect.top,
      right: rect.right - hostRect.left,
      bottom: rect.bottom - hostRect.top,
      width: rect.width,
      height: rect.height
    };
  });

  const contentWidth = Math.max(...canvasRects.map((rect) => rect.right));
  const contentHeight = Math.max(...canvasRects.map((rect) => rect.bottom));
  const output = makeCanvas(contentWidth * EXPORT_SCALE, contentHeight * EXPORT_SCALE);
  const context = output.getContext("2d");

  if (!context) {
    throw new Error("Unable to prepare the PDF canvas.");
  }

  fillWhite(context, output.width, output.height);

  for (const rect of canvasRects) {
    context.drawImage(
      rect.canvas,
      rect.left * EXPORT_SCALE,
      rect.top * EXPORT_SCALE,
      rect.width * EXPORT_SCALE,
      rect.height * EXPORT_SCALE
    );
  }

  return output;
}

async function paginateScore(scoreCanvas: HTMLCanvasElement): Promise<PdfImage[]> {
  const contentWidthPt = A4_WIDTH_PT - PAGE_MARGIN_PT * 2;
  const contentHeightPt = A4_HEIGHT_PT - PAGE_MARGIN_PT * 2;
  const sourcePageHeight = Math.floor(scoreCanvas.width * (contentHeightPt / contentWidthPt));
  const pages: PdfImage[] = [];

  for (let sourceY = 0; sourceY < scoreCanvas.height; sourceY += sourcePageHeight) {
    const sliceHeight = Math.min(sourcePageHeight, scoreCanvas.height - sourceY);
    const pageCanvas = makeCanvas(scoreCanvas.width, sliceHeight);
    const context = pageCanvas.getContext("2d");

    if (!context) {
      throw new Error("Unable to prepare a PDF page.");
    }

    fillWhite(context, pageCanvas.width, pageCanvas.height);
    context.drawImage(
      scoreCanvas,
      0,
      sourceY,
      scoreCanvas.width,
      sliceHeight,
      0,
      0,
      pageCanvas.width,
      pageCanvas.height
    );

    pages.push({
      bytes: await canvasToJpegBytes(pageCanvas),
      width: pageCanvas.width,
      height: pageCanvas.height,
      displayHeightPt: (sliceHeight / scoreCanvas.width) * contentWidthPt
    });
  }

  return pages;
}

function asciiBytes(value: string) {
  return new TextEncoder().encode(value);
}

function concatBytes(parts: Uint8Array[]) {
  const totalLength = parts.reduce((total, part) => total + part.length, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;

  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }

  return output;
}

function buildPdf(images: PdfImage[]): Blob {
  const objectParts: Uint8Array[] = [];
  const offsets: number[] = [0];
  const header = asciiBytes("%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");
  let cursor = header.length;
  const objects: Uint8Array[] = [];
  const catalogObject = 1;
  const pagesObject = 2;
  const firstPageObject = 3;
  const firstContentObject = firstPageObject + images.length;
  const firstImageObject = firstContentObject + images.length;

  function addObject(id: number, parts: Array<string | Uint8Array>) {
    const header = asciiBytes(`${id} 0 obj\n`);
    const footer = asciiBytes("\nendobj\n");
    const body = parts.map((part) => (typeof part === "string" ? asciiBytes(part) : part));
    const bytes = concatBytes([header, ...body, footer]);
    objects[id] = bytes;
  }

  const pageIds = images.map((_, index) => firstPageObject + index);
  addObject(catalogObject, [`<< /Type /Catalog /Pages ${pagesObject} 0 R >>`]);
  addObject(pagesObject, [`<< /Type /Pages /Count ${images.length} /Kids [${pageIds.map((id) => `${id} 0 R`).join(" ")}] >>`]);

  images.forEach((image, index) => {
    const pageId = firstPageObject + index;
    const contentId = firstContentObject + index;
    const imageId = firstImageObject + index;
    const imageName = `Im${index + 1}`;
    const contentWidthPt = A4_WIDTH_PT - PAGE_MARGIN_PT * 2;
    const imageY = A4_HEIGHT_PT - PAGE_MARGIN_PT - image.displayHeightPt;
    const commands = [
      "q",
      `${contentWidthPt.toFixed(2)} 0 0 ${image.displayHeightPt.toFixed(2)} ${PAGE_MARGIN_PT} ${imageY.toFixed(2)} cm`,
      `/${imageName} Do`,
      "Q"
    ].join("\n");

    addObject(pageId, [
      `<< /Type /Page /Parent ${pagesObject} 0 R /MediaBox [0 0 ${A4_WIDTH_PT} ${A4_HEIGHT_PT}] `,
      `/Resources << /XObject << /${imageName} ${imageId} 0 R >> >> /Contents ${contentId} 0 R >>`
    ]);
    addObject(contentId, [`<< /Length ${commands.length} >>\nstream\n${commands}\nendstream`]);
    addObject(imageId, [
      `<< /Type /XObject /Subtype /Image /Width ${image.width} /Height ${image.height} `,
      `/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${image.bytes.length} >>\nstream\n`,
      image.bytes,
      "\nendstream"
    ]);
  });

  objectParts.push(header);
  for (let id = 1; id < objects.length; id += 1) {
    offsets[id] = cursor;
    objectParts.push(objects[id]);
    cursor += objects[id].length;
  }

  const xrefOffset = cursor;
  const xrefRows = ["xref", `0 ${objects.length}`, "0000000000 65535 f "];
  for (let id = 1; id < objects.length; id += 1) {
    xrefRows.push(`${offsets[id].toString().padStart(10, "0")} 00000 n `);
  }
  xrefRows.push(
    "trailer",
    `<< /Size ${objects.length} /Root ${catalogObject} 0 R >>`,
    "startxref",
    String(xrefOffset),
    "%%EOF"
  );
  objectParts.push(asciiBytes(`${xrefRows.join("\n")}\n`));

  return new Blob([concatBytes(objectParts)], { type: "application/pdf" });
}

export async function exportScoreElementToPdf(host: HTMLElement): Promise<Blob> {
  const scoreCanvas = collectScoreCanvas(host);
  const pages = await paginateScore(scoreCanvas);
  return buildPdf(pages);
}

export async function savePdfBlob(blob: Blob, fileName: string) {
  const picker = (window as SavePickerWindow).showSaveFilePicker;

  if (picker) {
    const handle = await picker({
      suggestedName: fileName,
      types: [
        {
          description: "PDF document",
          accept: { "application/pdf": [".pdf"] }
        }
      ]
    });
    const writable = await handle.createWritable();
    await writable.write(blob);
    await writable.close();
    return;
  }

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

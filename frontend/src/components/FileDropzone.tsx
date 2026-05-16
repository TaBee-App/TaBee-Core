import { Upload } from "lucide-react";
import { useState } from "react";
import { formatBytes } from "../lib/format";

interface FileDropzoneProps {
  file: File | null;
  onFileChange: (file: File) => void;
}

export function FileDropzone({ file, onFileChange }: FileDropzoneProps) {
  const [dragging, setDragging] = useState(false);
  const meta = file ? `${formatBytes(file.size)} / ${file.type || "audio"}` : "WAV, MP3, OGG, or M4A";

  function handleFile(candidate?: File) {
    if (!candidate) return;
    onFileChange(candidate);
  }

  return (
    <label
      className={`dropzone${dragging ? " dragging" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(event) => {
        event.preventDefault();
        setDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        handleFile(event.dataTransfer.files?.[0]);
      }}
    >
      <input
        type="file"
        accept="audio/*,.wav,.mp3,.ogg,.m4a"
        onChange={(event) => handleFile(event.target.files?.[0])}
      />
      <span className="drop-icon">
        <Upload size={24} />
      </span>
      <span className="drop-title">{file ? file.name : "Choose or drop audio"}</span>
      <span className="drop-meta">{meta}</span>
    </label>
  );
}

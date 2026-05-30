import { Sparkles } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { generateTab } from "../api/tabeeApi";
import { FileDropzone } from "../components/FileDropzone";
import { errorMessage } from "../lib/errors";
import { upsertTab } from "../lib/tabStore";
import type { Instrument } from "../types/tab";

export function GeneratePage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [instrument, setInstrument] = useState<Instrument>("bass");
  const [tuning, setTuning] = useState<"EADG" | "BEADG">("EADG");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function selectFile(nextFile: File) {
    setFile(nextFile);
    setError("");
    setTitle((current) => current || nextFile.name.replace(/\.[^.]+$/, "").replace(/[-_]+/g, " "));
  }

  async function submit() {
    if (!file) {
      setError("Choose an audio file first.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await generateTab({
        file,
        title: title || "Generated tab",
        instrument,
        tuning
      });

      upsertTab(response);
      navigate(`/tabs/${response.id}`);
    } catch (caught) {
      setError(errorMessage(caught, "Error during processing."));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="generator-layout">
      <section className="workspace-panel">
        <p className="eyebrow">Generate</p>
        <h2>Turn a recording into a playable tab.</h2>
        <p className="panel-copy">
          Upload a bass or guitar recording. The current backend contract returns AlphaTex, and the frontend renders it as
          a product-ready tab view.
        </p>

        <FileDropzone file={file} onFileChange={selectFile} />

        <label className="field">
          <span>Title</span>
          <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Generated tab" />
        </label>

        <label className="field">
          <span>Instrument</span>
          <select value={instrument} onChange={(event) => setInstrument(event.target.value as Instrument)}>
            <option value="bass">Bass</option>
            <option value="guitar">Guitar</option>
          </select>
        </label>

        <label className="field">
          <span>Tuning</span>
          <select value={tuning} onChange={(event) => setTuning(event.target.value as "EADG" | "BEADG")}>
            <option value="EADG">4-string standard bass (E A D G)</option>
            <option value="BEADG">5-string bass (B E A D G)</option>
          </select>
        </label>

        {error ? <div className="form-error">{error}</div> : null}

        <button className="btn primary full" disabled={loading} onClick={submit}>
          <Sparkles size={18} />
          {loading ? "Generating..." : "Generate tab"}
        </button>
      </section>

      <section className="process-panel">
        <h2>Pipeline preview</h2>
        <div className="process-list">
          <div className="process-step active">
            <span>1</span>
            <div>
              <strong>Audio upload</strong>
              <p>Capture the source recording and instrument choice.</p>
            </div>
          </div>
          <div className="process-step">
            <span>2</span>
            <div>
              <strong>Note detection</strong>
              <p>TaBee-Core estimates pitch and onset positions.</p>
            </div>
          </div>
          <div className="process-step">
            <span>3</span>
            <div>
              <strong>Tab generation</strong>
              <p>Detected notes are assigned to playable frets and strings.</p>
            </div>
          </div>
          <div className="process-step">
            <span>4</span>
            <div>
              <strong>Product view</strong>
              <p>AlphaTex is rendered as an interactive tab with playback controls.</p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

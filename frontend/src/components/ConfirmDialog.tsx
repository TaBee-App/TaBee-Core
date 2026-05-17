import { AlertTriangle, X } from "lucide-react";

interface ConfirmDialogProps {
  title: string;
  message: string;
  confirmLabel?: string;
  loading?: boolean;
  tone?: "danger" | "default";
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  title,
  message,
  confirmLabel = "Confirm",
  loading = false,
  tone = "default",
  onConfirm,
  onCancel
}: ConfirmDialogProps) {
  return (
    <div className="modal-backdrop" role="presentation">
      <div className="modal-panel confirm-panel" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
        <div className="section-header">
          <div className="confirm-title-row">
            <AlertTriangle size={20} />
            <div>
              <h2 id="confirm-title">{title}</h2>
              <p>{message}</p>
            </div>
          </div>
          <button className="icon-btn subtle" type="button" title="Close" disabled={loading} onClick={onCancel}>
            <X size={17} />
          </button>
        </div>
        <div className="confirm-actions">
          <button className="btn ghost" disabled={loading} onClick={onCancel}>
            Cancel
          </button>
          <button className={`btn ${tone === "danger" ? "ghost danger" : "primary"}`} disabled={loading} onClick={onConfirm}>
            {loading ? "Working..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

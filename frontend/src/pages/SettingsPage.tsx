import { Moon, Sun, Trash2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { deleteMe } from "../api/authApi";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { errorMessage } from "../lib/errors";
import { applyTheme, getStoredTheme, type ThemeMode } from "../lib/theme";

export function SettingsPage() {
  const navigate = useNavigate();
  const [theme, setTheme] = useState<ThemeMode>(() => getStoredTheme());
  const [deletePassword, setDeletePassword] = useState("");
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [deletingAccount, setDeletingAccount] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);

  function changeTheme(nextTheme: ThemeMode) {
    setTheme(nextTheme);
    applyTheme(nextTheme);
  }

  function requestDeleteAccount() {
    if (deleteConfirmation !== "delete my account") {
      setDeleteError('Type "delete my account" to confirm.');
      return;
    }
    if (!deletePassword.trim()) {
      setDeleteError("Current password is required.");
      return;
    }
    setDeleteError("");
    setDeleteConfirmOpen(true);
  }

  async function submitDeleteAccount() {
    setDeletingAccount(true);
    setDeleteError("");
    try {
      await deleteMe({
        currentPassword: deletePassword,
        confirmation: deleteConfirmation
      });
      navigate("/auth", { replace: true });
    } catch (caught) {
      setDeleteConfirmOpen(false);
      setDeleteError(errorMessage(caught, "Could not delete account."));
    } finally {
      setDeletingAccount(false);
    }
  }

  return (
    <main className="settings-page">
      <section className="settings-panel">
        <p className="eyebrow">Settings</p>
        <h2>Viewer preferences</h2>
        <p>
          This page is intentionally small for the first frontend milestone. The settings that matter next are theme,
          default instrument, default tuning, cursor style, and notation scale.
        </p>

        <div className="setting-row">
          <div>
            <strong>Theme</strong>
            <span>Choose the interface contrast that feels best while reading tabs.</span>
          </div>
          <div className="theme-toggle" aria-label="Theme mode">
            <button className={theme === "dark" ? "active" : ""} onClick={() => changeTheme("dark")}>
              <Moon size={16} />
              Dark
            </button>
            <button className={theme === "light" ? "active" : ""} onClick={() => changeTheme("light")}>
              <Sun size={16} />
              Light
            </button>
          </div>
        </div>

        <div className="setting-row">
          <div>
            <strong>Default backend</strong>
            <span>Development proxy points `/api` to `127.0.0.1:8080`.</span>
          </div>
          <code>vite.config.ts</code>
        </div>

        <div className="setting-row">
          <div>
            <strong>Renderer</strong>
            <span>AlphaTab renders generated AlphaTex as tablature.</span>
          </div>
          <code>@coderline/alphatab</code>
        </div>

        <div className="setting-row danger-zone">
          <div>
            <strong>Delete account</strong>
            <span>
              Permanently removes your profile, tabs, playlists, follows, saved playlists, and favorites.
            </span>
          </div>
          <div className="danger-form">
            {deleteError ? <div className="form-error">{deleteError}</div> : null}
            <label className="field">
              <span>Current password</span>
              <input
                type="password"
                value={deletePassword}
                onChange={(event) => setDeletePassword(event.target.value)}
              />
            </label>
            <label className="field">
              <span>Confirmation</span>
              <input
                value={deleteConfirmation}
                placeholder="delete my account"
                onChange={(event) => setDeleteConfirmation(event.target.value)}
              />
            </label>
            <button className="btn ghost danger" disabled={deletingAccount} onClick={requestDeleteAccount}>
              <Trash2 size={18} />
              {deletingAccount ? "Deleting..." : "Delete account"}
            </button>
          </div>
        </div>
      </section>
      {deleteConfirmOpen ? (
        <ConfirmDialog
          title="Delete account permanently?"
          message="This removes your account, tabs, playlists, follows, saved playlists, and favorites. This cannot be undone."
          confirmLabel="Delete account"
          loading={deletingAccount}
          tone="danger"
          onCancel={() => setDeleteConfirmOpen(false)}
          onConfirm={submitDeleteAccount}
        />
      ) : null}
    </main>
  );
}

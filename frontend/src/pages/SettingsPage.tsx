import { ChevronDown, Moon, Save, Sun, Trash2 } from "lucide-react";
import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { deleteMe, updateMe } from "../api/authApi";
import { getCurrentUser } from "../api/authSession";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { errorMessage } from "../lib/errors";
import { passwordPolicyError, passwordRules } from "../lib/passwordPolicy";
import { applyTheme, getStoredTheme, type ThemeMode } from "../lib/theme";

export function SettingsPage() {
  const navigate = useNavigate();
  const currentUser = getCurrentUser();
  const [theme, setTheme] = useState<ThemeMode>(() => getStoredTheme());
  const [profileForm, setProfileForm] = useState({
    username: currentUser?.username || "",
    email: currentUser?.email || "",
    fullName: currentUser?.fullName || "",
    currentPassword: "",
    password: ""
  });
  const [savingProfile, setSavingProfile] = useState(false);
  const [profileError, setProfileError] = useState("");
  const [profileSaved, setProfileSaved] = useState(false);
  const [editProfileOpen, setEditProfileOpen] = useState(false);
  const [deletePassword, setDeletePassword] = useState("");
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [deletingAccount, setDeletingAccount] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deleteAccountOpen, setDeleteAccountOpen] = useState(false);
  const newPasswordRules = passwordRules(profileForm.password, {
    username: profileForm.username,
    email: profileForm.email,
    fullName: profileForm.fullName
  });

  function changeTheme(nextTheme: ThemeMode) {
    setTheme(nextTheme);
    applyTheme(nextTheme);
  }

  async function submitProfile(event: FormEvent) {
    event.preventDefault();
    const username = profileForm.username.trim();
    const email = profileForm.email.trim();
    const fullName = profileForm.fullName.trim();
    const currentPassword = profileForm.currentPassword.trim();
    const password = profileForm.password.trim();

    if (!username || !email) {
      setProfileError("Username and email are required.");
      return;
    }
    if (!currentPassword) {
      setProfileError("Current password is required to update your profile.");
      return;
    }
    if (password) {
      const policyError = passwordPolicyError(password, { username, email, fullName });
      if (policyError) {
        setProfileError(policyError);
        return;
      }
    }

    setSavingProfile(true);
    setProfileError("");
    setProfileSaved(false);
    try {
      const updatedUser = await updateMe({
        currentPassword,
        username,
        email,
        fullName,
        ...(password ? { password } : {})
      });
      setProfileForm({
        username: updatedUser.username,
        email: updatedUser.email,
        fullName: updatedUser.fullName || "",
        currentPassword: "",
        password: ""
      });
      setProfileSaved(true);
    } catch (caught) {
      setProfileError(errorMessage(caught, "Could not update profile."));
    } finally {
      setSavingProfile(false);
    }
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
        <h2>Account and preferences</h2>
        <p>
          Manage your profile, reading preference, and account safety from one place.
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

        <section className={`settings-accordion${editProfileOpen ? " open" : ""}`}>
          <button
            className="settings-accordion-trigger"
            type="button"
            aria-expanded={editProfileOpen}
            onClick={() => setEditProfileOpen((current) => !current)}
          >
            <span>
              <strong>Edit profile</strong>
              <small>Update username, email, display name, or password.</small>
            </span>
            <ChevronDown size={18} />
          </button>

          {editProfileOpen ? (
            <form className="settings-form-panel" onSubmit={submitProfile}>
              {profileError ? <div className="form-error">{profileError}</div> : null}
              {profileSaved ? <div className="form-success">Profile updated.</div> : null}

              <div className="settings-form-grid">
                <label className="field">
                  <span>Username</span>
                  <input
                    value={profileForm.username}
                    onChange={(event) => setProfileForm((current) => ({ ...current, username: event.target.value }))}
                  />
                </label>
                <label className="field">
                  <span>Email</span>
                  <input
                    type="email"
                    value={profileForm.email}
                    onChange={(event) => setProfileForm((current) => ({ ...current, email: event.target.value }))}
                  />
                </label>
              </div>

              <label className="field">
                <span>Full name</span>
                <input
                  value={profileForm.fullName}
                  onChange={(event) => setProfileForm((current) => ({ ...current, fullName: event.target.value }))}
                />
              </label>

              <div className="settings-form-grid">
                <label className="field">
                  <span>Current password</span>
                  <input
                    type="password"
                    value={profileForm.currentPassword}
                    placeholder="Required to save changes"
                    onChange={(event) =>
                      setProfileForm((current) => ({ ...current, currentPassword: event.target.value }))
                    }
                  />
                </label>
                <label className="field">
                  <span>New password</span>
                  <input
                    type="password"
                    value={profileForm.password}
                    placeholder="Leave blank to keep current password"
                    minLength={8}
                    onChange={(event) => setProfileForm((current) => ({ ...current, password: event.target.value }))}
                  />
                </label>
              </div>

              {profileForm.password ? (
                <ul className="password-rules">
                  {newPasswordRules.map((rule) => (
                    <li className={rule.passed ? "passed" : ""} key={rule.id}>
                      {rule.label}
                    </li>
                  ))}
                </ul>
              ) : null}

              <button className="btn primary" disabled={savingProfile} type="submit">
                <Save size={18} />
                {savingProfile ? "Saving..." : "Save profile"}
              </button>
            </form>
          ) : null}
        </section>

        <section className={`settings-accordion danger-accordion${deleteAccountOpen ? " open" : ""}`}>
          <button
            className="settings-accordion-trigger"
            type="button"
            aria-expanded={deleteAccountOpen}
            onClick={() => setDeleteAccountOpen((current) => !current)}
          >
            <span>
              <strong>Delete account</strong>
              <small>Permanently removes your profile, tabs, playlists, follows, saved playlists, and favorites.</small>
            </span>
            <ChevronDown size={18} />
          </button>

          {deleteAccountOpen ? (
            <div className="danger-form">
              {deleteError ? <div className="form-error">{deleteError}</div> : null}
              <label className="field">
                <span>Password</span>
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
          ) : null}
        </section>
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

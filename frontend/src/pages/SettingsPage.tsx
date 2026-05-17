import { ChevronDown, ImagePlus, Moon, Pencil, Save, Sun, Trash2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  confirmEmailUpdate,
  deleteMe,
  removeProfileImage,
  requestEmailUpdateCode,
  updateMe,
  updateProfileImage
} from "../api/authApi";
import { getCurrentUser } from "../api/authSession";
import { Avatar } from "../components/Avatar";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { errorMessage } from "../lib/errors";
import { passwordPolicyError, passwordRules } from "../lib/passwordPolicy";
import { applyTheme, getStoredTheme, type ThemeMode } from "../lib/theme";
import type { UserProfile } from "../types/auth";

type EditableField = "username" | "email" | "fullName" | "password";

export function SettingsPage() {
  const navigate = useNavigate();
  const [userProfile, setUserProfile] = useState<UserProfile | null>(() => getCurrentUser());
  const [theme, setTheme] = useState<ThemeMode>(() => getStoredTheme());
  const [editProfileOpen, setEditProfileOpen] = useState(false);
  const [activeEdit, setActiveEdit] = useState<EditableField | null>(null);
  const [draftValue, setDraftValue] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [profileError, setProfileError] = useState("");
  const [profileSaved, setProfileSaved] = useState("");
  const [uploadingProfileImage, setUploadingProfileImage] = useState(false);
  const [emailCodeOpen, setEmailCodeOpen] = useState(false);
  const [emailVerificationCode, setEmailVerificationCode] = useState("");
  const [emailVerificationHint, setEmailVerificationHint] = useState("");
  const [deletePassword, setDeletePassword] = useState("");
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [deletingAccount, setDeletingAccount] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deleteAccountOpen, setDeleteAccountOpen] = useState(false);

  const passwordRulesForDraft = passwordRules(draftValue, {
    username: userProfile?.username,
    email: userProfile?.email,
    fullName: userProfile?.fullName || undefined
  });

  function changeTheme(nextTheme: ThemeMode) {
    setTheme(nextTheme);
    applyTheme(nextTheme);
  }

  function startEdit(field: EditableField) {
    setProfileError("");
    setProfileSaved("");
    setActiveEdit(field);
    setCurrentPassword("");
    setConfirmPassword("");
    setEmailVerificationCode("");
    setEmailVerificationHint("");
    if (field === "username") setDraftValue(userProfile?.username || "");
    if (field === "email") setDraftValue(userProfile?.email || "");
    if (field === "fullName") setDraftValue(userProfile?.fullName || "");
    if (field === "password") setDraftValue("");
  }

  function cancelEdit() {
    setActiveEdit(null);
    setDraftValue("");
    setCurrentPassword("");
    setConfirmPassword("");
    setEmailCodeOpen(false);
    setEmailVerificationCode("");
    setEmailVerificationHint("");
  }

  async function changeProfileImage(file: File | undefined) {
    if (!file) return;
    if (!["image/png", "image/jpeg"].includes(file.type)) {
      setProfileError("Profile photo must be a PNG or JPEG image.");
      return;
    }

    setUploadingProfileImage(true);
    setProfileError("");
    setProfileSaved("");
    try {
      const updatedUser = await updateProfileImage(file);
      setUserProfile(updatedUser);
      setProfileSaved("Profile photo updated.");
    } catch (caught) {
      setProfileError(errorMessage(caught, "Could not update profile photo."));
    } finally {
      setUploadingProfileImage(false);
    }
  }

  async function deleteProfileImage() {
    setUploadingProfileImage(true);
    setProfileError("");
    setProfileSaved("");
    try {
      const updatedUser = await removeProfileImage();
      setUserProfile(updatedUser);
      setProfileSaved("Profile photo removed.");
    } catch (caught) {
      setProfileError(errorMessage(caught, "Could not remove profile photo."));
    } finally {
      setUploadingProfileImage(false);
    }
  }

  async function saveField() {
    if (!activeEdit || !userProfile) return;
    const trimmedValue = draftValue.trim();
    const trimmedPassword = currentPassword.trim();

    if (!trimmedPassword) {
      setProfileError("Current password is required.");
      return;
    }
    if (activeEdit !== "fullName" && !trimmedValue) {
      setProfileError("This field cannot be empty.");
      return;
    }
    if (activeEdit === "password") {
      if (draftValue !== confirmPassword) {
        setProfileError("Passwords do not match.");
        return;
      }
      const policyError = passwordPolicyError(draftValue, {
        username: userProfile.username,
        email: userProfile.email,
        fullName: userProfile.fullName || undefined
      });
      if (policyError) {
        setProfileError(policyError);
        return;
      }
    }

    setSavingProfile(true);
    setProfileError("");
    setProfileSaved("");
    try {
      if (activeEdit === "email") {
        const response = await requestEmailUpdateCode({ email: trimmedValue, currentPassword: trimmedPassword });
        setEmailVerificationHint(response.devCode ? `Development code: ${response.devCode}` : response.message);
        setEmailVerificationCode("");
        setEmailCodeOpen(true);
        return;
      }

      const updatedUser = await updateMe({
        currentPassword: trimmedPassword,
        ...(activeEdit === "username" ? { username: trimmedValue } : {}),
        ...(activeEdit === "fullName" ? { fullName: trimmedValue } : {}),
        ...(activeEdit === "password" ? { password: draftValue } : {})
      });
      setUserProfile(updatedUser);
      setProfileSaved("Profile updated.");
      cancelEdit();
    } catch (caught) {
      setProfileError(errorMessage(caught, "Could not update profile."));
    } finally {
      setSavingProfile(false);
    }
  }

  async function confirmEmailChange() {
    if (!activeEdit || activeEdit !== "email") return;
    setSavingProfile(true);
    setProfileError("");
    try {
      const updatedUser = await confirmEmailUpdate({
        email: draftValue.trim(),
        currentPassword: currentPassword.trim(),
        verificationCode: emailVerificationCode
      });
      setUserProfile(updatedUser);
      setProfileSaved("Email updated.");
      cancelEdit();
    } catch (caught) {
      setProfileError(errorMessage(caught, "Could not verify email."));
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
        <p>Manage your profile, reading preference, and account safety from one place.</p>

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
              <small>Edit one profile field at a time.</small>
            </span>
            <ChevronDown size={18} />
          </button>

          {editProfileOpen ? (
            <div className="settings-form-panel">
              {profileError ? <div className="form-error">{profileError}</div> : null}
              {profileSaved ? <div className="form-success">{profileSaved}</div> : null}

              <div className="profile-photo-editor">
                <Avatar src={userProfile?.profileImageUrl} label={userProfile?.username} size="lg" />
                <div>
                  <strong>Profile photo</strong>
                  <span>PNG or JPEG only.</span>
                  <div className="profile-photo-actions">
                    <label className={`btn ghost ${uploadingProfileImage ? "disabled" : ""}`}>
                      <ImagePlus size={17} />
                      Upload photo
                      <input
                        type="file"
                        accept="image/png,image/jpeg"
                        disabled={uploadingProfileImage}
                        onChange={(event) => changeProfileImage(event.target.files?.[0])}
                      />
                    </label>
                    {userProfile?.profileImageUrl ? (
                      <button className="btn ghost danger" type="button" disabled={uploadingProfileImage} onClick={deleteProfileImage}>
                        <Trash2 size={17} />
                        Remove
                      </button>
                    ) : null}
                  </div>
                </div>
              </div>

              <EditableProfileRow
                label="Username"
                value={userProfile?.username || ""}
                active={activeEdit === "username"}
                help={usernameHelp(userProfile)}
                draftValue={draftValue}
                currentPassword={currentPassword}
                saving={savingProfile}
                onEdit={() => startEdit("username")}
                onDraftChange={setDraftValue}
                onPasswordChange={setCurrentPassword}
                onCancel={cancelEdit}
                onSave={saveField}
              />
              <EditableProfileRow
                label="Email"
                value={userProfile?.email || ""}
                active={activeEdit === "email"}
                help="Changing email requires a code sent to the new address."
                draftValue={draftValue}
                currentPassword={currentPassword}
                saving={savingProfile}
                type="email"
                onEdit={() => startEdit("email")}
                onDraftChange={setDraftValue}
                onPasswordChange={setCurrentPassword}
                onCancel={cancelEdit}
                onSave={saveField}
              />
              <EditableProfileRow
                label="Full name"
                value={userProfile?.fullName || "Not set"}
                active={activeEdit === "fullName"}
                draftValue={draftValue}
                currentPassword={currentPassword}
                saving={savingProfile}
                onEdit={() => startEdit("fullName")}
                onDraftChange={setDraftValue}
                onPasswordChange={setCurrentPassword}
                onCancel={cancelEdit}
                onSave={saveField}
              />
              <EditableProfileRow
                label="Password"
                value="••••••••"
                active={activeEdit === "password"}
                draftValue={draftValue}
                currentPassword={currentPassword}
                confirmPassword={confirmPassword}
                saving={savingProfile}
                type="password"
                passwordRules={passwordRulesForDraft}
                onEdit={() => startEdit("password")}
                onDraftChange={setDraftValue}
                onPasswordChange={setCurrentPassword}
                onConfirmPasswordChange={setConfirmPassword}
                onCancel={cancelEdit}
                onSave={saveField}
              />
            </div>
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

      {emailCodeOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div className="modal-panel confirm-panel" role="dialog" aria-modal="true" aria-labelledby="email-code-title">
            <div className="section-header">
              <div>
                <h2 id="email-code-title">Verify new email</h2>
                <p>Enter the 6-digit code sent to {draftValue}.</p>
              </div>
            </div>
            <label className="field">
              <span>Verification code</span>
              <input
                value={emailVerificationCode}
                onChange={(event) => setEmailVerificationCode(event.target.value.replace(/\D/g, "").slice(0, 6))}
                inputMode="numeric"
                pattern="\d{6}"
                placeholder="000000"
              />
            </label>
            {emailVerificationHint ? <div className="form-note">{emailVerificationHint}</div> : null}
            {profileError ? <div className="form-error">{profileError}</div> : null}
            <div className="confirm-actions">
              <button className="btn ghost" disabled={savingProfile} onClick={cancelEdit}>
                Cancel
              </button>
              <button className="btn primary" disabled={savingProfile || emailVerificationCode.length !== 6} onClick={confirmEmailChange}>
                {savingProfile ? "Checking..." : "Verify email"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

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

function EditableProfileRow({
  label,
  value,
  active,
  help,
  draftValue,
  currentPassword,
  confirmPassword,
  saving,
  type = "text",
  passwordRules,
  onEdit,
  onDraftChange,
  onPasswordChange,
  onConfirmPasswordChange,
  onCancel,
  onSave
}: {
  label: string;
  value: string;
  active: boolean;
  help?: string;
  draftValue: string;
  currentPassword: string;
  confirmPassword?: string;
  saving: boolean;
  type?: "text" | "email" | "password";
  passwordRules?: Array<{ id: string; label: string; passed: boolean }>;
  onEdit: () => void;
  onDraftChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onConfirmPasswordChange?: (value: string) => void;
  onCancel: () => void;
  onSave: () => void;
}) {
  return (
    <div className={`editable-profile-row${active ? " editing" : ""}`}>
      <div className="editable-profile-summary">
        <div>
          <strong>{label}</strong>
          <span>{value}</span>
          {help ? <small>{help}</small> : null}
        </div>
        {!active ? (
          <button className="icon-btn subtle" type="button" title={`Edit ${label}`} onClick={onEdit}>
            <Pencil size={16} />
          </button>
        ) : null}
      </div>

      {active ? (
        <div className="editable-profile-editor">
          <label className="field">
            <span>{type === "password" ? "New password" : label}</span>
            <input
              type={type}
              value={draftValue}
              onChange={(event) => onDraftChange(event.target.value)}
              autoComplete={type === "password" ? "new-password" : undefined}
            />
          </label>
          {type === "password" ? (
            <>
              <ul className="password-rules">
                {(passwordRules || []).map((rule) => (
                  <li className={rule.passed ? "passed" : ""} key={rule.id}>
                    {rule.label}
                  </li>
                ))}
              </ul>
              <label className="field">
                <span>Confirm new password</span>
                <input
                  type="password"
                  value={confirmPassword || ""}
                  onChange={(event) => onConfirmPasswordChange?.(event.target.value)}
                  autoComplete="new-password"
                />
              </label>
            </>
          ) : null}
          <label className="field">
            <span>Current password</span>
            <input
              type="password"
              value={currentPassword}
              onChange={(event) => onPasswordChange(event.target.value)}
              autoComplete="current-password"
            />
          </label>
          <div className="editable-profile-actions">
            <button className="btn ghost" type="button" disabled={saving} onClick={onCancel}>
              Cancel
            </button>
            <button className="btn primary" type="button" disabled={saving} onClick={onSave}>
              <Save size={17} />
              {saving ? "Saving..." : "Save"}
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function usernameHelp(userProfile: UserProfile | null) {
  if (!userProfile?.usernameUpdatedAt) {
    return "Username can be changed once per month.";
  }
  const nextDate = new Date(userProfile.usernameUpdatedAt);
  nextDate.setMonth(nextDate.getMonth() + 1);
  return `Next username change after ${nextDate.toLocaleDateString()}.`;
}

import { Compass, LogOut, Pencil, Save, Search, Settings, Sparkles, UserRound, X } from "lucide-react";
import { FormEvent, useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { logout, updateMe } from "../api/authApi";
import { getCurrentUser } from "../api/authSession";
import type { UserProfile } from "../types/auth";

export function AppShell() {
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(() => getCurrentUser());
  const [accountOpen, setAccountOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);
  const [profileError, setProfileError] = useState("");
  const [profileForm, setProfileForm] = useState({
    username: currentUser?.username || "",
    email: currentUser?.email || "",
    fullName: currentUser?.fullName || "",
    currentPassword: "",
    password: ""
  });

  function signOut() {
    logout();
    navigate("/auth", { replace: true });
  }

  function openProfileEditor() {
    setProfileForm({
      username: currentUser?.username || "",
      email: currentUser?.email || "",
      fullName: currentUser?.fullName || "",
      currentPassword: "",
      password: ""
    });
    setProfileError("");
    setAccountOpen(false);
    setEditOpen(true);
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

    setSavingProfile(true);
    setProfileError("");
    try {
      const updatedUser = await updateMe({
        currentPassword,
        username,
        email,
        fullName,
        ...(password ? { password } : {})
      });
      setCurrentUser(updatedUser);
      setEditOpen(false);
      setProfileForm((current) => ({ ...current, currentPassword: "", password: "" }));
    } catch (caught) {
      setProfileError(caught instanceof Error ? caught.message : "Could not update profile.");
    } finally {
      setSavingProfile(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink to="/" className="brand" aria-label="TaBee dashboard">
            <div className="brand-mark">T</div>
            <div>
              <h1 className="brand-title">TaBee</h1>
              <p className="brand-subtitle">Audio to playable tablature</p>
            </div>
          </NavLink>

          <nav className="nav-tabs" aria-label="Main navigation">
            <NavLink to="/generate">
              <Sparkles size={18} />
              Generate
            </NavLink>
            <NavLink to="/search">
              <Search size={18} />
              Search
            </NavLink>
            <NavLink to="/discover">
              <Compass size={18} />
              Discovery
            </NavLink>
          </nav>

          <div className="account-menu">
            <button className="account-trigger" title="Account" onClick={() => setAccountOpen((current) => !current)}>
              <UserRound size={17} />
              <span>{currentUser?.username || "Profile"}</span>
            </button>
            {accountOpen ? (
              <div className="account-dropdown">
                <NavLink to="/profile" onClick={() => setAccountOpen(false)}>
                  <UserRound size={17} />
                  Profile
                </NavLink>
                <button onClick={openProfileEditor}>
                  <Pencil size={17} />
                  Edit profile
                </button>
                <NavLink to="/settings" onClick={() => setAccountOpen(false)}>
                  <Settings size={17} />
                  Settings
                </NavLink>
                <button onClick={signOut}>
                  <LogOut size={17} />
                  Log out
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </header>

      <Outlet />

      {editOpen ? (
        <div className="modal-backdrop" role="presentation">
          <form className="modal-panel" onSubmit={submitProfile}>
            <div className="section-header">
              <div>
                <h2>Edit profile</h2>
                <p>Update the account details shown across TaBee.</p>
              </div>
              <button className="icon-btn subtle" type="button" title="Close" onClick={() => setEditOpen(false)}>
                <X size={17} />
              </button>
            </div>

            {profileError ? <div className="form-error">{profileError}</div> : null}

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
            <label className="field">
              <span>Full name</span>
              <input
                value={profileForm.fullName}
                onChange={(event) => setProfileForm((current) => ({ ...current, fullName: event.target.value }))}
              />
            </label>
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
                onChange={(event) => setProfileForm((current) => ({ ...current, password: event.target.value }))}
              />
            </label>
            <button className="btn primary full" disabled={savingProfile} type="submit">
              <Save size={18} />
              {savingProfile ? "Saving..." : "Save profile"}
            </button>
          </form>
        </div>
      ) : null}
    </div>
  );
}

import { Library, ListMusic, LogOut, Settings, Sparkles, UserRound } from "lucide-react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { logout } from "../api/authApi";
import { getCurrentUser } from "../api/authSession";

export function AppShell() {
  const navigate = useNavigate();
  const currentUser = getCurrentUser();

  function signOut() {
    logout();
    navigate("/auth", { replace: true });
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
            <NavLink to="/" end>
              <Library size={18} />
              Library
            </NavLink>
            <NavLink to="/generate">
              <Sparkles size={18} />
              Generate
            </NavLink>
            <NavLink to="/playlists">
              <ListMusic size={18} />
              Playlists
            </NavLink>
            <NavLink to="/settings">
              <Settings size={18} />
              Settings
            </NavLink>
          </nav>

          <div className="account-pill">
            <NavLink to="/profile" className="profile-chip" title="Profile">
              <UserRound size={17} />
              <span>{currentUser?.username || "Profile"}</span>
            </NavLink>
            <button className="icon-btn subtle" title="Logout" onClick={signOut}>
              <LogOut size={17} />
            </button>
          </div>
        </div>
      </header>

      <Outlet />
    </div>
  );
}

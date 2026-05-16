import { Compass, LogOut, Search, Settings, Sparkles, UserRound } from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { logout } from "../api/authApi";
import { getCurrentUser } from "../api/authSession";

export function AppShell() {
  const navigate = useNavigate();
  const currentUser = getCurrentUser();
  const [accountOpen, setAccountOpen] = useState(false);

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
    </div>
  );
}

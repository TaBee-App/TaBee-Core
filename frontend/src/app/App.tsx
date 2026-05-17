import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "../components/AppShell";
import { RequireAuth } from "../components/RequireAuth";
import { AuthPage } from "../pages/AuthPage";
import { DashboardPage } from "../pages/DashboardPage";
import { DiscoveryPage } from "../pages/DiscoveryPage";
import { GeneratePage } from "../pages/GeneratePage";
import { PlaylistDetailPage } from "../pages/PlaylistDetailPage";
import { ProfilePage } from "../pages/ProfilePage";
import { PublicUserPage } from "../pages/PublicUserPage";
import { SearchPage } from "../pages/SearchPage";
import { SettingsPage } from "../pages/SettingsPage";
import { TabViewerPage } from "../pages/TabViewerPage";
import { UserConnectionsPage } from "../pages/UserConnectionsPage";

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/auth" element={<AuthPage />} />
        <Route element={<RequireAuth />}>
          <Route element={<AppShell />}>
            <Route index element={<DashboardPage />} />
            <Route path="/generate" element={<GeneratePage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/discover" element={<DiscoveryPage />} />
            <Route path="/playlists" element={<Navigate to="/discover" replace />} />
            <Route path="/playlists/:playlistId" element={<PlaylistDetailPage />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="/users/:userId" element={<PublicUserPage />} />
            <Route path="/users/:userId/:kind" element={<UserConnectionsPage />} />
            <Route path="/tabs/:tabId" element={<TabViewerPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

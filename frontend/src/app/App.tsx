import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "../components/AppShell";
import { RequireAuth } from "../components/RequireAuth";
import { AuthPage } from "../pages/AuthPage";
import { DashboardPage } from "../pages/DashboardPage";
import { GeneratePage } from "../pages/GeneratePage";
import { PlaylistsPage } from "../pages/PlaylistsPage";
import { PlaylistDetailPage } from "../pages/PlaylistDetailPage";
import { ProfilePage } from "../pages/ProfilePage";
import { PublicUserPage } from "../pages/PublicUserPage";
import { SettingsPage } from "../pages/SettingsPage";
import { TabViewerPage } from "../pages/TabViewerPage";

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/auth" element={<AuthPage />} />
        <Route element={<RequireAuth />}>
          <Route element={<AppShell />}>
            <Route index element={<DashboardPage />} />
            <Route path="/generate" element={<GeneratePage />} />
            <Route path="/playlists" element={<PlaylistsPage />} />
            <Route path="/playlists/:playlistId" element={<PlaylistDetailPage />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="/users/:userId" element={<PublicUserPage />} />
            <Route path="/tabs/:tabId" element={<TabViewerPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

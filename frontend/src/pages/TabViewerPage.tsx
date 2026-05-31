import { FileDown, Pencil, Plus, Save, Star, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  addTabToPlaylist,
  deleteGeneratedTab,
  favoriteTab,
  getGeneratedTab,
  getPublicGeneratedTab,
  listPlaylists,
  unfavoriteTab,
  updateGeneratedTab
} from "../api/tabeeApi";
import { PlayerBar } from "../components/PlayerBar";
import { Avatar } from "../components/Avatar";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { TabRenderer } from "../components/TabRenderer";
import { errorMessage } from "../lib/errors";
import { slugify } from "../lib/format";
import { demoTab } from "../lib/demoTab";
import { getTab, removeTab, upsertTab } from "../lib/tabStore";
import type { GeneratedTab, PlaylistResponse } from "../types/tab";

export function TabViewerPage() {
  const { tabId } = useParams();
  const navigate = useNavigate();
  const [tab, setTab] = useState<GeneratedTab | null>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [looping, setLooping] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [speed, setSpeed] = useState(100);
  const [renderKey, setRenderKey] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [metadata, setMetadata] = useState({
    title: "",
    artist: "",
    tuning: "",
    tempo: ""
  });
  const [playlists, setPlaylists] = useState<PlaylistResponse[]>([]);
  const [selectedPlaylistId, setSelectedPlaylistId] = useState("");
  const [playlistSaving, setPlaylistSaving] = useState(false);
  const [favoriteSaving, setFavoriteSaving] = useState(false);
  const [deletingTab, setDeletingTab] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);

  useEffect(() => {
    let active = true;

    if (tabId === "demo") {
      upsertTab(demoTab);
      setTab(demoTab);
      setMetadata(toMetadataForm(demoTab));
      setLoading(false);
      setError("");
      return;
    }

    async function loadTab() {
      const cachedTab = tabId ? getTab(tabId) : null;
      setTab(cachedTab);
      if (cachedTab) {
        setMetadata(toMetadataForm(cachedTab));
      }
      setLoading(Boolean(tabId));
      setError("");

      if (!tabId) {
        setLoading(false);
        return;
      }

      try {
        let backendTab: GeneratedTab;
        try {
          backendTab = await getGeneratedTab(tabId);
        } catch {
          backendTab = await getPublicGeneratedTab(tabId);
        }
        if (!active) return;
        upsertTab(backendTab);
        setTab(backendTab);
        setMetadata(toMetadataForm(backendTab));
      } catch (caught) {
        if (!active) return;
        if (!cachedTab) {
          setError(errorMessage(caught, "Could not load tab."));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadTab();
    return () => {
      active = false;
    };
  }, [tabId]);

  useEffect(() => {
    let active = true;

    async function loadPlaylists() {
      try {
        const response = await listPlaylists();
        if (!active) return;
        setPlaylists(response);
        setSelectedPlaylistId((current) => current || (response[0] ? String(response[0].id) : ""));
      } catch {
        if (active) {
          setPlaylists([]);
        }
      }
    }

    loadPlaylists();
    return () => {
      active = false;
    };
  }, []);

  const meta = useMemo(() => {
    if (!tab) return "Generate or open a recent tab to enable playback";
    const tempo = tab.tempo ? `${tab.tempo} BPM` : "tempo auto";
    return `${tab.instrument} / ${tempo} / ${speed}% speed${looping ? " / loop on" : ""}`;
  }, [looping, speed, tab]);

  function downloadPdf() {
    if (!tab) return;

    setPlaying(false);
    const previousTitle = document.title;
    document.title = `${slugify(tab.title)}-tab`;

    const restoreTitle = () => {
      document.title = previousTitle;
      window.removeEventListener("afterprint", restoreTitle);
    };

    window.addEventListener("afterprint", restoreTitle);
    window.setTimeout(() => {
      window.print();
      window.setTimeout(restoreTitle, 1000);
    }, 100);
  }

  function startEditing() {
    if (!tab || tab.id === "demo") return;
    setMetadata(toMetadataForm(tab));
    setEditing(true);
    setError("");
  }

  function cancelEditing() {
    if (tab) {
      setMetadata(toMetadataForm(tab));
    }
    setEditing(false);
    setError("");
  }

  async function saveMetadata() {
    if (!tab || tab.id === "demo") return;

    const nextTitle = metadata.title.trim();
    if (!nextTitle) {
      setError("Title cannot be empty.");
      return;
    }

    const nextTempo = metadata.tempo.trim() ? Number(metadata.tempo) : null;
    if (nextTempo !== null && (!Number.isFinite(nextTempo) || nextTempo <= 0)) {
      setError("Tempo must be a positive number.");
      return;
    }

    setSaving(true);
    setError("");
    try {
      const updatedTab = await updateGeneratedTab(tab.id, {
        title: nextTitle,
        artist: nullableText(metadata.artist),
        tuning: nullableText(metadata.tuning),
        estimatedTempo: nextTempo
      });
      upsertTab(updatedTab);
      setTab(updatedTab);
      setMetadata(toMetadataForm(updatedTab));
      setEditing(false);
      setRenderKey((current) => current + 1);
    } catch (caught) {
      setError(errorMessage(caught, "Could not save tab details."));
    } finally {
      setSaving(false);
    }
  }

  async function addToPlaylist() {
    if (!tab || tab.id === "demo" || !selectedPlaylistId) return;

    setPlaylistSaving(true);
    setError("");
    try {
      const playlist = await addTabToPlaylist(Number(selectedPlaylistId), tab.id);
      setPlaylists((current) => current.map((item) => (item.id === playlist.id ? playlist : item)));
    } catch (caught) {
      setError(errorMessage(caught, "Could not add tab to playlist."));
    } finally {
      setPlaylistSaving(false);
    }
  }

  async function toggleFavorite() {
    if (!tab || tab.id === "demo" || tab.createdByCurrentUser) return;

    setFavoriteSaving(true);
    setError("");
    try {
      const updatedTab = tab.favoritedByCurrentUser ? await unfavoriteTab(tab.id) : await favoriteTab(tab.id);
      upsertTab(updatedTab);
      setTab(updatedTab);
    } catch (caught) {
      setError(errorMessage(caught, "Could not update favorite."));
    } finally {
      setFavoriteSaving(false);
    }
  }

  async function deleteTab() {
    if (!tab || tab.id === "demo" || !tab.createdByCurrentUser) return;

    setDeletingTab(true);
    setError("");
    try {
      await deleteGeneratedTab(tab.id);
      removeTab(tab.id);
      navigate("/profile");
    } catch (caught) {
      setDeleteConfirmOpen(false);
      setError(errorMessage(caught, "Could not delete tab."));
    } finally {
      setDeletingTab(false);
    }
  }

  if (!tab && loading) {
    return (
      <main className="viewer-page">
        <div className="empty-panel">
          <h2>Loading tab</h2>
          <p>Preparing your saved tab.</p>
        </div>
      </main>
    );
  }

  if (!tab) {
    return (
      <main className="viewer-page">
        <div className="empty-panel">
          <h2>Tab not found</h2>
          <p>{error || "Generate a new one or open the demo."}</p>
          <Link to="/generate" className="btn primary">Generate tab</Link>
        </div>
      </main>
    );
  }

  return (
    <>
      <main className="viewer-page">
        <section className="score-panel">
          <div className="score-header">
            <div>
              {editing ? (
                <div className="metadata-editor">
                  <label className="metadata-field wide">
                    <span>Title</span>
                    <input
                      value={metadata.title}
                      onChange={(event) => setMetadata((current) => ({ ...current, title: event.target.value }))}
                    />
                  </label>
                  <label className="metadata-field">
                    <span>Artist</span>
                    <input
                      value={metadata.artist}
                      onChange={(event) => setMetadata((current) => ({ ...current, artist: event.target.value }))}
                    />
                  </label>
                  <label className="metadata-field short">
                    <span>Tuning</span>
                    <input
                      value={metadata.tuning}
                      onChange={(event) => setMetadata((current) => ({ ...current, tuning: event.target.value }))}
                      placeholder="BEADG"
                    />
                  </label>
                  <label className="metadata-field short">
                    <span>BPM</span>
                    <input
                      inputMode="numeric"
                      value={metadata.tempo}
                      onChange={(event) => setMetadata((current) => ({ ...current, tempo: event.target.value }))}
                    />
                  </label>
                </div>
              ) : (
                <>
                  <h2 className="score-title">{tab.title}</h2>
                  <p className="score-subtitle">
                    {[tab.fileName, tab.artist, tab.tuning].filter(Boolean).join(" / ")}
                  </p>
                  {tab.ownerUserId && tab.ownerUsername ? (
                    <Link className="creator-pill" to={`/users/${tab.ownerUserId}`}>
                      <Avatar src={tab.ownerProfileImageUrl} label={tab.ownerUsername} size="sm" />
                      {tab.createdByCurrentUser ? "Created by me" : `Created by @${tab.ownerUsername}`}
                    </Link>
                  ) : null}
                  <span className="favorite-count-pill">
                    <Star size={15} />
                    {formatFavoriteCount(tab.favoriteCount || 0)}
                  </span>
                </>
              )}
            </div>
            <div className="score-actions">
              {editing ? (
                <>
                  <button className="btn ghost" disabled={saving} onClick={cancelEditing}>
                    <X size={17} />
                    Cancel
                  </button>
                  <button className="btn primary" disabled={saving} onClick={saveMetadata}>
                    <Save size={17} />
                    {saving ? "Saving..." : "Save"}
                  </button>
                </>
              ) : tab.id !== "demo" ? (
                <>
                  <div className="score-primary-actions">
                    {playlists.length ? (
                      <div className="playlist-adder">
                        <select
                          aria-label="Playlist"
                          value={selectedPlaylistId}
                          onChange={(event) => setSelectedPlaylistId(event.target.value)}
                        >
                          {playlists.map((playlist) => (
                            <option key={playlist.id} value={playlist.id}>
                              {playlist.name}
                            </option>
                          ))}
                        </select>
                        <button
                          className="icon-btn primary"
                          title={playlistSaving ? "Adding to playlist" : "Add to playlist"}
                          disabled={playlistSaving || !selectedPlaylistId}
                          onClick={addToPlaylist}
                        >
                          <Plus size={18} />
                        </button>
                      </div>
                    ) : null}
                    {!tab.createdByCurrentUser ? (
                      <button
                        className={`icon-btn subtle${tab.favoritedByCurrentUser ? " active" : ""}`}
                        title={tab.favoritedByCurrentUser ? "Favorited" : "Favorite"}
                        disabled={favoriteSaving}
                        onClick={toggleFavorite}
                      >
                        <Star size={17} />
                      </button>
                    ) : null}
                  </div>
                  <div className="score-tool-actions">
                    <button className="icon-btn subtle" title="Edit details" onClick={startEditing}>
                      <Pencil size={17} />
                    </button>
                    {tab.createdByCurrentUser ? (
                      <button
                        className="icon-btn subtle danger"
                        title="Delete tab"
                        disabled={deletingTab}
                        onClick={() => setDeleteConfirmOpen(true)}
                      >
                        <Trash2 size={17} />
                      </button>
                    ) : null}
                    <button className="icon-btn subtle" title="Download PDF" onClick={downloadPdf}>
                      <FileDown size={17} />
                    </button>
                  </div>
                </>
              ) : (
                <div className="score-tool-actions">
                  <button className="icon-btn subtle" title="Download PDF" onClick={downloadPdf}>
                    <FileDown size={17} />
                  </button>
                </div>
              )}
            </div>
          </div>
          {error ? <div className="form-error">{error}</div> : null}
          <TabRenderer
            key={`${tab.id}-${renderKey}`}
            tab={tab}
            playing={playing}
            looping={looping}
            autoScroll={autoScroll}
            speed={speed}
            enableSynthPlayback
            onReadyChange={setReady}
            onPlayingChange={setPlaying}
          />
        </section>
      </main>

      <PlayerBar
        title={tab.title}
        meta={meta}
        ready={ready}
        playing={playing}
        looping={looping}
        autoScroll={autoScroll}
        speed={speed}
        audioUrl={tab.audioUrl}
        onTogglePlay={() => setPlaying((current) => !current)}
        onLoopChange={setLooping}
        onAutoScrollChange={setAutoScroll}
        onSpeedChange={setSpeed}
      />
      {deleteConfirmOpen && tab ? (
        <ConfirmDialog
          title="Delete tab?"
          message={`Delete "${tab.title}"? This removes it from your account and playlists.`}
          confirmLabel="Delete tab"
          loading={deletingTab}
          tone="danger"
          onCancel={() => setDeleteConfirmOpen(false)}
          onConfirm={deleteTab}
        />
      ) : null}
    </>
  );
}

function toMetadataForm(tab: GeneratedTab) {
  return {
    title: tab.title,
    artist: tab.artist || "",
    tuning: tab.tuning || "",
    tempo: tab.tempo == null ? "" : String(tab.tempo)
  };
}

function nullableText(value: string) {
  const trimmed = value.trim();
  return trimmed || null;
}

function formatFavoriteCount(count: number) {
  return `${count} ${count === 1 ? "Favorite" : "Favorites"}`;
}

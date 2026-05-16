import { Star, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getPlaylist, removeTabFromPlaylist, savePlaylist, unsavePlaylist } from "../api/tabeeApi";
import type { PlaylistResponse } from "../types/tab";

export function PlaylistDetailPage() {
  const { playlistId } = useParams();
  const [playlist, setPlaylist] = useState<PlaylistResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadPlaylist() {
      if (!playlistId) return;
      setLoading(true);
      setError("");
      try {
        const response = await getPlaylist(playlistId);
        if (active) {
          setPlaylist(response);
        }
      } catch (caught) {
        if (active) {
          setError(caught instanceof Error ? caught.message : "Could not load playlist.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadPlaylist();
    return () => {
      active = false;
    };
  }, [playlistId]);

  async function toggleSave() {
    if (!playlist || playlist.createdByCurrentUser) return;
    try {
      const updated = playlist.savedByCurrentUser ? await unsavePlaylist(playlist.id) : await savePlaylist(playlist.id);
      setPlaylist(updated);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not update saved playlist.");
    }
  }

  async function removeTab(tabId: number) {
    if (!playlist || !playlist.createdByCurrentUser) return;
    try {
      setPlaylist(await removeTabFromPlaylist(playlist.id, tabId));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not remove tab.");
    }
  }

  if (!playlist && loading) {
    return (
      <main className="page-grid">
        <div className="empty-panel">
          <h2>Loading playlist</h2>
          <p>Fetching playlist details.</p>
        </div>
      </main>
    );
  }

  if (!playlist) {
    return (
      <main className="page-grid">
        <div className="empty-panel">
          <h2>Playlist not found</h2>
          <p>{error || "This playlist is unavailable."}</p>
        </div>
      </main>
    );
  }

  return (
    <main className="page-grid">
      <section className="library-hero">
        <div>
          <p className="eyebrow">{playlist.createdByCurrentUser ? "Created by you" : "Created by"}</p>
          <h2>{playlist.name}</h2>
          <p>
            {!playlist.createdByCurrentUser ? (
              <>
                <Link className="inline-link" to={`/users/${playlist.ownerUserId}`}>@{playlist.ownerUsername}</Link>
                {" / "}
              </>
            ) : null}
            {playlist.description || `${playlist.tabs.length} tabs in this playlist`}
          </p>
        </div>
        {!playlist.createdByCurrentUser ? (
          <button className={`btn ${playlist.savedByCurrentUser ? "ghost" : "primary"}`} onClick={toggleSave}>
            <Star size={18} />
            {playlist.savedByCurrentUser ? "Saved" : "Save playlist"}
          </button>
        ) : null}
      </section>

      {error ? <div className="form-error">{error}</div> : null}

      <section className="library-panel">
        <div className="section-header">
          <div>
            <h2>Tabs</h2>
            <p>{playlist.tabs.length} tabs in this playlist</p>
          </div>
        </div>

        <div className="tab-list">
          {playlist.tabs.map((tab) => (
            <article className="tab-card" key={tab.tabId}>
              <Link to={`/tabs/${tab.tabId}`}>
                <span className="tab-card-title">{tab.title}</span>
                <span className="tab-card-meta">
                  {tab.artist || "TaBee tab"}
                  {!playlist.createdByCurrentUser ? ` / owner #${tab.ownerUserId}` : ""}
                </span>
              </Link>
              {playlist.createdByCurrentUser ? (
                <button className="icon-btn subtle" title="Remove from playlist" onClick={() => removeTab(tab.tabId)}>
                  <Trash2 size={17} />
                </button>
              ) : null}
            </article>
          ))}
        </div>

        {!playlist.tabs.length ? (
          <div className="empty-panel compact">
            <h3>No tabs yet</h3>
            <p>This playlist is empty.</p>
          </div>
        ) : null}
      </section>
    </main>
  );
}

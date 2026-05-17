import { Star, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { deletePlaylist, getPlaylist, removeTabFromPlaylist, savePlaylist, unsavePlaylist } from "../api/tabeeApi";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { errorMessage } from "../lib/errors";
import type { PlaylistResponse } from "../types/tab";

export function PlaylistDetailPage() {
  const { playlistId } = useParams();
  const navigate = useNavigate();
  const [playlist, setPlaylist] = useState<PlaylistResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [deletingPlaylist, setDeletingPlaylist] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [removingTabId, setRemovingTabId] = useState<number | null>(null);
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
          setError(errorMessage(caught, "Could not load playlist."));
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
      setError(errorMessage(caught, "Could not update saved playlist."));
    }
  }

  async function removeTab(tabId: number) {
    if (!playlist || !playlist.createdByCurrentUser) return;
    setRemovingTabId(tabId);
    try {
      setPlaylist(await removeTabFromPlaylist(playlist.id, tabId));
    } catch (caught) {
      setError(errorMessage(caught, "Could not remove tab."));
    } finally {
      setRemovingTabId(null);
    }
  }

  async function removePlaylist() {
    if (!playlist || !playlist.createdByCurrentUser) return;

    setDeletingPlaylist(true);
    try {
      await deletePlaylist(playlist.id);
      navigate("/profile");
    } catch (caught) {
      setDeleteConfirmOpen(false);
      setError(errorMessage(caught, "Could not delete playlist."));
    } finally {
      setDeletingPlaylist(false);
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
        ) : (
          <button className="btn ghost danger" onClick={() => setDeleteConfirmOpen(true)}>
            <Trash2 size={18} />
            Delete playlist
          </button>
        )}
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
                <button
                  className="icon-btn subtle"
                  title="Remove from playlist"
                  disabled={removingTabId === tab.tabId}
                  onClick={() => removeTab(tab.tabId)}
                >
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
      {deleteConfirmOpen && playlist ? (
        <ConfirmDialog
          title="Delete playlist?"
          message={`Delete "${playlist.name}"? Tabs inside it will stay in your account.`}
          confirmLabel="Delete playlist"
          loading={deletingPlaylist}
          tone="danger"
          onCancel={() => setDeleteConfirmOpen(false)}
          onConfirm={removePlaylist}
        />
      ) : null}
    </main>
  );
}

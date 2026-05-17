import { ImagePlus, Star, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  deletePlaylist,
  getPlaylist,
  removePlaylistCover,
  removeTabFromPlaylist,
  savePlaylist,
  unsavePlaylist,
  updatePlaylistCover
} from "../api/tabeeApi";
import { Avatar } from "../components/Avatar";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { PlaylistCover } from "../components/PlaylistCover";
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
  const [updatingCover, setUpdatingCover] = useState(false);
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

  async function changeCover(file: File | undefined) {
    if (!playlist || !playlist.createdByCurrentUser || !file) return;
    if (!["image/png", "image/jpeg"].includes(file.type)) {
      setError("Playlist cover must be a PNG or JPEG image.");
      return;
    }

    setUpdatingCover(true);
    setError("");
    try {
      setPlaylist(await updatePlaylistCover(playlist.id, file));
    } catch (caught) {
      setError(errorMessage(caught, "Could not update playlist cover."));
    } finally {
      setUpdatingCover(false);
    }
  }

  async function deleteCover() {
    if (!playlist || !playlist.createdByCurrentUser) return;
    setUpdatingCover(true);
    setError("");
    try {
      setPlaylist(await removePlaylistCover(playlist.id));
    } catch (caught) {
      setError(errorMessage(caught, "Could not remove playlist cover."));
    } finally {
      setUpdatingCover(false);
    }
  }

  if (!playlist && loading) {
    return (
      <main className="page-grid">
        <div className="empty-panel">
          <h2>Loading playlist</h2>
          <p>Preparing the playlist page.</p>
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
        <div className="playlist-hero-cover">
          <PlaylistCover src={playlist.coverImageUrl} title={playlist.name} size="md" />
        </div>
        <div>
          <p className="eyebrow">{playlist.createdByCurrentUser ? "Created by you" : "Created by"}</p>
          <h2>{playlist.name}</h2>
          <p>
            {!playlist.createdByCurrentUser ? (
              <>
                <Link className="inline-link owner-inline" to={`/users/${playlist.ownerUserId}`}>
                  <Avatar src={playlist.ownerProfileImageUrl} label={playlist.ownerUsername} size="sm" />
                  @{playlist.ownerUsername}
                </Link>
                {" / "}
              </>
            ) : null}
            {playlist.description || `${playlist.tabs.length} tabs in this playlist`}
          </p>
          <div className="detail-stat-row">
            <span className="favorite-count-pill">
              <Star size={15} />
              {formatSaveCount(playlist.savedCount || 0)}
            </span>
            <span className="favorite-count-pill">Created {formatPlaylistDate(playlist.createdAt)}</span>
          </div>
        </div>
        <div className="playlist-detail-actions">
          {playlist.createdByCurrentUser ? (
            <>
              <div className="playlist-cover-tools">
                <label className={`btn ghost ${updatingCover ? "disabled" : ""}`}>
                  <ImagePlus size={18} />
                  Change cover
                  <input
                    type="file"
                    accept="image/png,image/jpeg"
                    disabled={updatingCover}
                    onChange={(event) => changeCover(event.target.files?.[0])}
                  />
                </label>
                {playlist.coverImageUrl ? (
                  <button className="btn ghost" disabled={updatingCover} onClick={deleteCover}>
                    Remove cover
                  </button>
                ) : null}
              </div>
              <button className="btn ghost danger" onClick={() => setDeleteConfirmOpen(true)}>
                <Trash2 size={18} />
                Delete playlist
              </button>
            </>
          ) : (
            <button className={`btn ${playlist.savedByCurrentUser ? "ghost" : "primary"}`} onClick={toggleSave}>
              <Star size={18} />
              {playlist.savedByCurrentUser ? "Saved" : "Save playlist"}
            </button>
          )}
        </div>
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

function formatPlaylistDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function formatSaveCount(count: number) {
  return `${count} ${count === 1 ? "Save" : "Saves"}`;
}

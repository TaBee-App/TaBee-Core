import { Music, Plus, Star, Trash2, UserRound, X } from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getPublicUser } from "../api/authApi";
import {
  addTabToPlaylist,
  createPlaylist,
  deleteGeneratedTab,
  deletePlaylist,
  listFavoriteTabs,
  listPlaylists,
  listSavedPlaylists,
  listTabs
} from "../api/tabeeApi";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { getCurrentUser } from "../api/authSession";
import { errorMessage } from "../lib/errors";
import { removeTab as removeCachedTab } from "../lib/tabStore";
import type { PublicUserProfile } from "../types/auth";
import type { GeneratedTab, PlaylistResponse } from "../types/tab";

export function ProfilePage() {
  const currentUser = getCurrentUser();
  const [profileSummary, setProfileSummary] = useState<PublicUserProfile | null>(null);
  const [tabs, setTabs] = useState<GeneratedTab[]>([]);
  const [favoriteTabs, setFavoriteTabs] = useState<GeneratedTab[]>([]);
  const [playlists, setPlaylists] = useState<PlaylistResponse[]>([]);
  const [savedPlaylists, setSavedPlaylists] = useState<PlaylistResponse[]>([]);
  const [playlistLayer, setPlaylistLayer] = useState<"created" | "saved">("created");
  const [tabLayer, setTabLayer] = useState<"created" | "favorited">("created");
  const [selectedPlaylists, setSelectedPlaylists] = useState<Record<string, string>>({});
  const [playlistName, setPlaylistName] = useState("");
  const [playlistDescription, setPlaylistDescription] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [creatingPlaylist, setCreatingPlaylist] = useState(false);
  const [savingTabId, setSavingTabId] = useState("");
  const [deletingId, setDeletingId] = useState("");
  const [confirmAction, setConfirmAction] = useState<
    | { type: "playlist"; playlist: PlaylistResponse }
    | { type: "tab"; tab: GeneratedTab }
    | null
  >(null);
  const [error, setError] = useState("");

  const playlistTabIds = useMemo(() => {
    return new Set(playlists.flatMap((playlist) => playlist.tabs.map((tab) => String(tab.tabId))));
  }, [playlists]);

  const unplaylistedTabs = useMemo(() => {
    return tabs.filter((tab) => !playlistTabIds.has(tab.id));
  }, [playlistTabIds, tabs]);

  useEffect(() => {
    let active = true;

    async function loadProfileData() {
      setLoading(true);
      setError("");
      try {
        const [nextTabs, nextFavoriteTabs, nextPlaylists, nextSavedPlaylists, nextProfileSummary] = await Promise.all([
          listTabs(),
          listFavoriteTabs(),
          listPlaylists(),
          listSavedPlaylists(),
          currentUser?.id ? getPublicUser(String(currentUser.id)) : Promise.resolve(null)
        ]);
        if (!active) return;
        setTabs(nextTabs);
        setFavoriteTabs(nextFavoriteTabs);
        setPlaylists(nextPlaylists);
        setSavedPlaylists(nextSavedPlaylists);
        setProfileSummary(nextProfileSummary);
        setSelectedPlaylists(defaultPlaylistSelections(nextTabs, nextPlaylists));
      } catch (caught) {
        if (active) {
          setError(errorMessage(caught, "Could not load profile."));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadProfileData();
    return () => {
      active = false;
    };
  }, []);

  async function addToPlaylist(tabId: string) {
    const playlistId = selectedPlaylists[tabId] || String(playlists[0]?.id || "");
    if (!playlistId) {
      setError("Create a playlist before adding tabs.");
      return;
    }

    setSavingTabId(tabId);
    setError("");
    try {
      const playlist = await addTabToPlaylist(Number(playlistId), tabId);
      setPlaylists((current) => current.map((item) => (item.id === playlist.id ? playlist : item)));
    } catch (caught) {
      setError(errorMessage(caught, "Could not add tab to playlist."));
    } finally {
      setSavingTabId("");
    }
  }

  async function submitPlaylist(event: FormEvent) {
    event.preventDefault();
    const name = playlistName.trim();
    if (!name) {
      setError("Playlist name is required.");
      return;
    }

    setCreatingPlaylist(true);
    setError("");
    try {
      const playlist = await createPlaylist({
        name,
        description: playlistDescription.trim() || null
      });
      const nextPlaylists = [playlist, ...playlists];
      setPlaylists(nextPlaylists);
      setSelectedPlaylists(defaultPlaylistSelections(tabs, nextPlaylists));
      setPlaylistName("");
      setPlaylistDescription("");
      setCreateOpen(false);
    } catch (caught) {
      setError(errorMessage(caught, "Could not create playlist."));
    } finally {
      setCreatingPlaylist(false);
    }
  }

  async function removePlaylist(playlist: PlaylistResponse) {
    setDeletingId(`playlist-${playlist.id}`);
    setError("");
    try {
      await deletePlaylist(playlist.id);
      const nextPlaylists = playlists.filter((item) => item.id !== playlist.id);
      setPlaylists(nextPlaylists);
      setSelectedPlaylists(defaultPlaylistSelections(tabs, nextPlaylists));
      setConfirmAction(null);
    } catch (caught) {
      setError(errorMessage(caught, "Could not delete playlist."));
    } finally {
      setDeletingId("");
    }
  }

  async function removeCreatedTab(tab: GeneratedTab) {
    setDeletingId(`tab-${tab.id}`);
    setError("");
    try {
      await deleteGeneratedTab(tab.id);
      removeCachedTab(tab.id);
      setTabs((current) => current.filter((item) => item.id !== tab.id));
      setPlaylists((current) =>
        current.map((playlist) => ({
          ...playlist,
          tabs: playlist.tabs.filter((playlistTab) => String(playlistTab.tabId) !== tab.id)
        }))
      );
      setSelectedPlaylists((current) => {
        const next = { ...current };
        delete next[tab.id];
        return next;
      });
      setConfirmAction(null);
    } catch (caught) {
      setError(errorMessage(caught, "Could not delete tab."));
    } finally {
      setDeletingId("");
    }
  }

  return (
    <main className="profile-page">
      <section className="profile-hero">
        <div className="profile-identity">
          <div className="profile-avatar">
            <UserRound size={28} />
          </div>
          <div className="profile-title-block">
            <p className="eyebrow">Profile</p>
            <h2>{currentUser?.fullName || currentUser?.username || "TaBee user"}</h2>
            <p>@{currentUser?.username || "profile"} / {currentUser?.email || "Signed in to TaBee"}</p>
            {profileSummary ? (
              <div className="social-counts">
                <Link to={`/users/${profileSummary.id}/followers`}>
                  <strong>{profileSummary.followerCount}</strong>
                  <span>Followers</span>
                </Link>
                <Link to={`/users/${profileSummary.id}/following`}>
                  <strong>{profileSummary.followingCount}</strong>
                  <span>Following</span>
                </Link>
              </div>
            ) : null}
          </div>
        </div>
        <div className="profile-stats">
          <span>
            <strong>{tabs.length}</strong>
            <small>Tabs</small>
          </span>
          <span>
            <strong>{favoriteTabs.length}</strong>
            <small>Favorites</small>
          </span>
          <span>
            <strong>{playlists.length}</strong>
            <small>Created playlists</small>
          </span>
          <span>
            <strong>{savedPlaylists.length}</strong>
            <small>Saved playlists</small>
          </span>
        </div>
      </section>

      {error ? <div className="form-error">{error}</div> : null}

      <section className="profile-grid">
        <div className="profile-panel">
          <div className="section-header">
            <div>
              <h2>My playlists</h2>
              <p>{loading ? "Loading playlists..." : `${playlists.length} created / ${savedPlaylists.length} saved`}</p>
            </div>
            <button className="btn primary" onClick={() => setCreateOpen(true)}>
              <Plus size={18} />
              New
            </button>
          </div>

          <div className="playlist-layer-grid">
            <button
              className={`playlist-layer-card${playlistLayer === "created" ? " active" : ""}`}
              onClick={() => setPlaylistLayer("created")}
            >
              <span>Created playlists</span>
              <strong>{playlists.length}</strong>
              <small>Playlists you own and manage</small>
            </button>
            <button
              className={`playlist-layer-card${playlistLayer === "saved" ? " active" : ""}`}
              onClick={() => setPlaylistLayer("saved")}
            >
              <span>Saved playlists</span>
              <strong>{savedPlaylists.length}</strong>
              <small>Playlists you saved from others</small>
            </button>
          </div>

          <div className="playlist-layer-content">
            <div className="section-header compact-header">
              <div>
                <h2>{playlistLayer === "created" ? "Created playlists" : "Saved playlists"}</h2>
                <p>
                  {playlistLayer === "created"
                    ? `${playlists.length} playlists created by you`
                    : `${savedPlaylists.length} saved collections`}
                </p>
              </div>
            </div>

            <div className="profile-list">
              {(playlistLayer === "created" ? playlists : savedPlaylists).map((playlist) => (
                <article className="profile-playlist-row" key={playlist.id}>
                  <Link to={`/playlists/${playlist.id}`}>
                    <span>{playlist.name}</span>
                    <small>
                      {playlistLayer === "created"
                        ? playlist.description || `${playlist.tabs.length} tabs`
                        : `by ${playlist.ownerUsername} / ${playlist.tabs.length} tabs`}
                    </small>
                  </Link>
                  <div className="profile-row-actions">
                    <strong>{playlist.tabs.length}</strong>
                    {playlistLayer === "created" ? (
                      <button
                        className="icon-btn subtle danger"
                        title="Delete playlist"
                        disabled={deletingId === `playlist-${playlist.id}`}
                        onClick={() => setConfirmAction({ type: "playlist", playlist })}
                      >
                        <Trash2 size={17} />
                      </button>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>

            {!loading && playlistLayer === "created" && !playlists.length ? (
              <div className="empty-panel compact">
                <h3>No created playlists yet</h3>
                <p>Create a playlist, then add your unplaylisted tabs to it.</p>
              </div>
            ) : null}

            {!loading && playlistLayer === "saved" && !savedPlaylists.length ? (
              <div className="empty-panel compact">
                <h3>No saved playlists yet</h3>
                <p>Use Discovery to save playlists created by other people.</p>
              </div>
            ) : null}
          </div>
        </div>

        <div className="profile-panel">
          <div className="section-header">
            <div>
              <h2>My tabs</h2>
              <p>{loading ? "Loading tabs..." : `${tabs.length} created / ${favoriteTabs.length} favorited`}</p>
            </div>
            <Music size={20} />
          </div>

          <div className="playlist-layer-grid">
            <button
              className={`playlist-layer-card${tabLayer === "created" ? " active" : ""}`}
              onClick={() => setTabLayer("created")}
            >
              <span>Created tabs</span>
              <strong>{tabs.length}</strong>
              <small>{unplaylistedTabs.length} still waiting for a playlist</small>
            </button>
            <button
              className={`playlist-layer-card${tabLayer === "favorited" ? " active" : ""}`}
              onClick={() => setTabLayer("favorited")}
            >
              <span>Favorited tabs</span>
              <strong>{favoriteTabs.length}</strong>
              <small>Tabs you liked from other users</small>
            </button>
          </div>

          <div className="playlist-layer-content">
            <div className="section-header compact-header">
              <div>
                <h2>{tabLayer === "created" ? "Created tabs" : "Favorited tabs"}</h2>
                <p>
                  {tabLayer === "created"
                    ? `${unplaylistedTabs.length} unplaylisted / ${tabs.length} total`
                    : `${favoriteTabs.length} tabs saved for later`}
                </p>
              </div>
              {tabLayer === "favorited" ? <Star size={18} /> : null}
            </div>

          <div className="profile-list">
            {(tabLayer === "created" ? tabs : favoriteTabs).map((tab) => (
              <article className="profile-tab-row" key={tab.id}>
                <Link to={`/tabs/${tab.id}`}>
                  <span>{tab.title}</span>
                  <small>
                    {tabLayer === "created"
                      ? `${tab.fileName} / ${tab.instrument}`
                      : `by ${tab.ownerUsername} / ${tab.artist || tab.instrument}`}
                  </small>
                </Link>
                {tabLayer === "created" ? (
                  <div className="profile-row-actions">
                    {playlistTabIds.has(tab.id) ? (
                      <span className="status-chip">In playlist</span>
                    ) : (
                      <>
                        <select
                          value={selectedPlaylists[tab.id] || ""}
                          onChange={(event) =>
                            setSelectedPlaylists((current) => ({ ...current, [tab.id]: event.target.value }))
                          }
                          disabled={!playlists.length}
                        >
                          {!playlists.length ? <option value="">No playlists</option> : null}
                          {playlists.map((playlist) => (
                            <option key={playlist.id} value={playlist.id}>
                              {playlist.name}
                            </option>
                          ))}
                        </select>
                        <button
                          className="icon-btn primary"
                          title="Add to playlist"
                          disabled={!playlists.length || savingTabId === tab.id}
                          onClick={() => addToPlaylist(tab.id)}
                        >
                          <Plus size={18} />
                        </button>
                      </>
                    )}
                    <button
                      className="icon-btn subtle danger"
                      title="Delete tab"
                      disabled={deletingId === `tab-${tab.id}`}
                      onClick={() => setConfirmAction({ type: "tab", tab })}
                    >
                      <Trash2 size={17} />
                    </button>
                  </div>
                ) : (
                  <Star className="favorite-inline-icon" size={18} />
                )}
              </article>
            ))}
          </div>

          {!loading && tabLayer === "created" && !tabs.length ? (
            <div className="empty-panel compact">
              <h3>No created tabs yet</h3>
              <p>Generate a tab, then organize it into playlists here.</p>
            </div>
          ) : null}

          {!loading && tabLayer === "favorited" && !favoriteTabs.length ? (
            <div className="empty-panel compact">
              <h3>No favorited tabs yet</h3>
              <p>Open tabs from Search or Discovery and favorite the ones you want to keep.</p>
            </div>
          ) : null}
          </div>
        </div>

      </section>

      {createOpen ? (
        <div className="modal-backdrop" role="presentation">
          <form className="modal-panel" onSubmit={submitPlaylist}>
            <div className="section-header">
              <div>
                <h2>Create playlist</h2>
                <p>Build a collection from your tabs.</p>
              </div>
              <button className="icon-btn subtle" type="button" title="Close" onClick={() => setCreateOpen(false)}>
                <X size={17} />
              </button>
            </div>
            <label className="field">
              <span>Name</span>
              <input
                value={playlistName}
                onChange={(event) => setPlaylistName(event.target.value)}
                placeholder="Practice set"
              />
            </label>
            <label className="field">
              <span>Description</span>
              <input
                value={playlistDescription}
                onChange={(event) => setPlaylistDescription(event.target.value)}
                placeholder="Optional notes"
              />
            </label>
            <button className="btn primary full" disabled={creatingPlaylist} type="submit">
              <Plus size={18} />
              {creatingPlaylist ? "Creating..." : "Create playlist"}
            </button>
          </form>
        </div>
      ) : null}
      {confirmAction ? (
        <ConfirmDialog
          title={confirmAction.type === "playlist" ? "Delete playlist?" : "Delete tab?"}
          message={
            confirmAction.type === "playlist"
              ? `Delete "${confirmAction.playlist.name}"? Tabs inside it will stay in your account.`
              : `Delete "${confirmAction.tab.title}"? This removes it from your account and playlists.`
          }
          confirmLabel={confirmAction.type === "playlist" ? "Delete playlist" : "Delete tab"}
          loading={
            confirmAction.type === "playlist"
              ? deletingId === `playlist-${confirmAction.playlist.id}`
              : deletingId === `tab-${confirmAction.tab.id}`
          }
          tone="danger"
          onCancel={() => setConfirmAction(null)}
          onConfirm={() =>
            confirmAction.type === "playlist"
              ? removePlaylist(confirmAction.playlist)
              : removeCreatedTab(confirmAction.tab)
          }
        />
      ) : null}
    </main>
  );
}

function defaultPlaylistSelections(tabs: GeneratedTab[], playlists: PlaylistResponse[]) {
  const firstPlaylist = playlists[0] ? String(playlists[0].id) : "";
  return Object.fromEntries(tabs.map((tab) => [tab.id, firstPlaylist]));
}

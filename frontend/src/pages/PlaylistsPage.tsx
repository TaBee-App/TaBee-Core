import { Clock, ListMusic, Music, Star } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  favoriteTab,
  listPlaylistArchive,
  listPublicTabs,
  savePlaylist,
  unfavoriteTab,
  unsavePlaylist
} from "../api/tabeeApi";
import { errorMessage } from "../lib/errors";
import type { GeneratedTab, PlaylistResponse } from "../types/tab";

export function PlaylistsPage() {
  const [playlists, setPlaylists] = useState<PlaylistResponse[]>([]);
  const [tabs, setTabs] = useState<GeneratedTab[]>([]);
  const [sortMode, setSortMode] = useState<"recent" | "mostTabs" | "favorites">("favorites");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const totalTabs = useMemo(
    () => playlists.reduce((count, playlist) => count + playlist.tabs.length, 0),
    [playlists]
  );

  const sortedPlaylists = useMemo(() => {
    return [...playlists].sort((left, right) => {
      if (sortMode === "mostTabs") return right.tabs.length - left.tabs.length;
      if (sortMode === "favorites") return Number(right.savedByCurrentUser) - Number(left.savedByCurrentUser);
      return new Date(right.createdAt).getTime() - new Date(left.createdAt).getTime();
    });
  }, [playlists, sortMode]);

  const sortedTabs = useMemo(() => {
    return [...tabs].sort((left, right) => {
      if (sortMode === "favorites") {
        return Number(right.favoritedByCurrentUser) - Number(left.favoritedByCurrentUser);
      }
      return Number(right.id) - Number(left.id);
    });
  }, [sortMode, tabs]);

  useEffect(() => {
    let active = true;

    async function load() {
      setLoading(true);
      setError("");
      try {
        const [nextPlaylists, nextTabs] = await Promise.all([listPlaylistArchive(), listPublicTabs()]);
        if (active) {
          setPlaylists(nextPlaylists);
          setTabs(nextTabs);
        }
      } catch (caught) {
        if (active) {
          setError(errorMessage(caught, "Could not load discovery."));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      active = false;
    };
  }, []);

  async function toggleSave(playlist: PlaylistResponse) {
    if (playlist.createdByCurrentUser) return;
    try {
      const updated = playlist.savedByCurrentUser
        ? await unsavePlaylist(playlist.id)
        : await savePlaylist(playlist.id);
      setPlaylists((current) => current.map((item) => (item.id === updated.id ? updated : item)));
    } catch (caught) {
      setError(errorMessage(caught, "Could not update saved playlist."));
    }
  }

  async function toggleFavorite(tab: GeneratedTab) {
    if (tab.createdByCurrentUser) return;
    try {
      const updated = tab.favoritedByCurrentUser ? await unfavoriteTab(tab.id) : await favoriteTab(tab.id);
      setTabs((current) => current.map((item) => (item.id === updated.id ? updated : item)));
    } catch (caught) {
      setError(errorMessage(caught, "Could not update favorite tab."));
    }
  }

  return (
    <main className="page-grid">
      <section className="library-hero">
        <div>
          <p className="eyebrow">Discovery</p>
          <h2>Browse tabs and playlist collections.</h2>
          <p>
            Explore recent and active community tabs alongside playlist collections.
          </p>
        </div>
        <div className="playlist-stat">
          <ListMusic size={22} />
          <span>{loading ? "Loading..." : `${tabs.length} tabs / ${playlists.length} playlists`}</span>
        </div>
      </section>

      <section className="library-panel">
        <div className="archive-tabs">
          <button className={sortMode === "favorites" ? "active" : ""} onClick={() => setSortMode("favorites")}>
            <Star size={17} />
            Most favorited
          </button>
          <button className={sortMode === "recent" ? "active" : ""} onClick={() => setSortMode("recent")}>
            <Clock size={17} />
            Most recent
          </button>
          <button className={sortMode === "mostTabs" ? "active" : ""} onClick={() => setSortMode("mostTabs")}>
            <ListMusic size={17} />
            Most tabs
          </button>
        </div>

        {error ? <div className="form-error">{error}</div> : null}

        <div className="playlist-feature-band">
          <div>
            <p className="eyebrow">Discovery</p>
            <h3>{sortMode === "favorites" ? "Favorited first" : sortMode === "mostTabs" ? "Largest collections" : "Most recent"}</h3>
            <p>Use Search when you know what you want. Use Discovery when you want to browse.</p>
          </div>
          <Star size={22} />
        </div>

        <div className="section-header discovery-section-header">
          <div>
            <h2>Tabs</h2>
            <p>{loading ? "Loading tabs..." : `${sortedTabs.length} public tabs`}</p>
          </div>
          <Music size={20} />
        </div>

        <div className="tab-list discovery-tab-list">
          {sortedTabs.map((tab) => (
            <article className="tab-card" key={tab.id}>
              <Link to={`/tabs/${tab.id}`}>
                <span className="tab-card-title">{tab.title}</span>
                <span className="tab-card-meta">
                  {tab.ownerUsername ? `@${tab.ownerUsername} / ` : ""}{tab.fileName} / {tab.instrument}
                </span>
              </Link>
              {!tab.createdByCurrentUser ? (
                <button
                  className={`icon-btn subtle${tab.favoritedByCurrentUser ? " active" : ""}`}
                  title={tab.favoritedByCurrentUser ? "Favorited" : "Favorite tab"}
                  onClick={() => toggleFavorite(tab)}
                >
                  <Star size={17} />
                </button>
              ) : null}
            </article>
          ))}
        </div>

        {!loading && !sortedTabs.length ? (
          <div className="empty-panel compact">
            <h3>No tabs to discover yet</h3>
            <p>As users generate public tabs, they will appear here.</p>
          </div>
        ) : null}

        <div className="section-header discovery-section-header">
          <div>
            <h2>Playlists</h2>
            <p>{loading ? "Loading playlists..." : `${sortedPlaylists.length} playlists / ${totalTabs} tabs`}</p>
          </div>
          <ListMusic size={20} />
        </div>

        <div className="playlist-grid">
          {sortedPlaylists.map((playlist) => (
            <article className="playlist-card" key={playlist.id}>
              <div className="playlist-card-header">
                <Link to={`/playlists/${playlist.id}`}>
                  <h3>{playlist.name}</h3>
                  <p>
                    by {playlist.ownerUsername} / {playlist.description || `${playlist.tabs.length} tabs`}
                  </p>
                </Link>
                {!playlist.createdByCurrentUser ? (
                  <button
                    className={`icon-btn subtle${playlist.savedByCurrentUser ? " active" : ""}`}
                    title={playlist.savedByCurrentUser ? "Saved" : "Save playlist"}
                    onClick={() => toggleSave(playlist)}
                  >
                    <Star size={17} />
                  </button>
                ) : null}
              </div>

              <div className="playlist-tabs">
                {playlist.tabs.map((tab) => (
                  <div className="playlist-tab-row" key={tab.tabId}>
                    <Link to={`/tabs/${tab.tabId}`}>
                      <span>{tab.title}</span>
                      <small>{tab.artist || "TaBee tab"}</small>
                    </Link>
                  </div>
                ))}
              </div>

              {!playlist.tabs.length ? <p className="playlist-empty">This saved playlist has no tabs yet.</p> : null}
            </article>
          ))}
        </div>

        {!loading && !sortedPlaylists.length ? (
          <div className="empty-panel">
            <h3>No discovery results</h3>
            <p>Create playlists from your profile page, then use Discovery to browse them.</p>
          </div>
        ) : null}
      </section>
    </main>
  );
}

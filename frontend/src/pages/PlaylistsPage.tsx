import { Clock, ListMusic, Search, Star } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { listPlaylistArchive, savePlaylist, unsavePlaylist } from "../api/tabeeApi";
import type { PlaylistResponse } from "../types/tab";

export function PlaylistsPage() {
  const [playlists, setPlaylists] = useState<PlaylistResponse[]>([]);
  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState<"recent" | "mostTabs" | "favorites">("recent");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const totalTabs = useMemo(
    () => playlists.reduce((count, playlist) => count + playlist.tabs.length, 0),
    [playlists]
  );

  const filteredPlaylists = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    const matching = normalized ? playlists.filter((playlist) =>
      `${playlist.name} ${playlist.description || ""} ${playlist.tabs.map((tab) => tab.title).join(" ")}`
        .toLowerCase()
        .includes(normalized)
    ) : playlists;

    return [...matching].sort((left, right) => {
      if (sortMode === "mostTabs") return right.tabs.length - left.tabs.length;
      if (sortMode === "favorites") return Number(right.savedByCurrentUser) - Number(left.savedByCurrentUser);
      return new Date(right.createdAt).getTime() - new Date(left.createdAt).getTime();
    });
  }, [playlists, query, sortMode]);

  useEffect(() => {
    let active = true;

    async function load() {
      setLoading(true);
      setError("");
      try {
          const response = await listPlaylistArchive();
        if (active) {
          setPlaylists(response);
        }
      } catch (caught) {
        if (active) {
          setError(caught instanceof Error ? caught.message : "Could not load playlists.");
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
      setError(caught instanceof Error ? caught.message : "Could not update saved playlist.");
    }
  }

  return (
    <main className="page-grid">
      <section className="library-hero">
        <div>
          <p className="eyebrow">Discovery</p>
          <h2>Browse playlist collections.</h2>
          <p>
            Explore recent, active, and saved playlist collections. Broader discovery modules can grow here later.
          </p>
        </div>
        <div className="playlist-stat">
          <ListMusic size={22} />
          <span>{loading ? "Loading..." : `${playlists.length} playlists / ${totalTabs} tabs`}</span>
        </div>
      </section>

      <section className="library-panel">
        <label className="search-box">
          <Search size={17} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter discovery playlists"
          />
        </label>

        <div className="archive-tabs">
          <button className={sortMode === "recent" ? "active" : ""} onClick={() => setSortMode("recent")}>
            <Clock size={17} />
            Most recent
          </button>
          <button className={sortMode === "mostTabs" ? "active" : ""} onClick={() => setSortMode("mostTabs")}>
            <ListMusic size={17} />
            Most tabs
          </button>
          <button className={sortMode === "favorites" ? "active" : ""} onClick={() => setSortMode("favorites")}>
            <Star size={17} />
            Most favorited
          </button>
        </div>

        {error ? <div className="form-error">{error}</div> : null}

        <div className="playlist-feature-band">
          <div>
            <p className="eyebrow">Later</p>
            <h3>Most favorited playlists</h3>
            <p>Favorites are not connected yet, but this section is reserved for playlist discovery.</p>
          </div>
          <Star size={22} />
        </div>

        <div className="playlist-grid">
          {filteredPlaylists.map((playlist) => (
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

        {!loading && !filteredPlaylists.length ? (
          <div className="empty-panel">
            <h3>No discovery results</h3>
            <p>Create playlists from your profile page, then use Discovery to browse them.</p>
          </div>
        ) : null}
      </section>
    </main>
  );
}

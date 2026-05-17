import { ListMusic, Music, Search, UserRound } from "lucide-react";
import type React from "react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { followUser, searchUsers, unfollowUser } from "../api/authApi";
import { listPlaylistArchive, listPublicTabs } from "../api/tabeeApi";
import type { PublicUserProfile } from "../types/auth";
import type { GeneratedTab, PlaylistResponse } from "../types/tab";

type SearchScope = "all" | "tabs" | "playlists" | "users";

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<SearchScope>("all");
  const [tabs, setTabs] = useState<GeneratedTab[]>([]);
  const [playlists, setPlaylists] = useState<PlaylistResponse[]>([]);
  const [users, setUsers] = useState<PublicUserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchingUsers, setSearchingUsers] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadSearchData() {
      setLoading(true);
      setError("");
      try {
        const [nextTabs, nextPlaylists] = await Promise.all([listPublicTabs(), listPlaylistArchive()]);
        if (!active) return;
        setTabs(nextTabs);
        setPlaylists(nextPlaylists);
      } catch (caught) {
        if (active) {
          setError(caught instanceof Error ? caught.message : "Could not load searchable content.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadSearchData();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    const normalized = query.trim();

    async function runUserSearch() {
      if ((scope !== "all" && scope !== "users") || normalized.length < 2) {
        setUsers([]);
        return;
      }

      setSearchingUsers(true);
      try {
        const results = await searchUsers(normalized);
        if (active) {
          setUsers(results);
        }
      } catch {
        if (active) {
          setUsers([]);
        }
      } finally {
        if (active) {
          setSearchingUsers(false);
        }
      }
    }

    const timer = window.setTimeout(runUserSearch, 250);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [query, scope]);

  const filteredTabs = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return tabs.slice(0, 8);
    return tabs.filter((tab) =>
      `${tab.title} ${tab.fileName} ${tab.instrument} ${tab.ownerUsername || ""}`.toLowerCase().includes(normalized)
    );
  }, [query, tabs]);

  const filteredPlaylists = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return playlists.slice(0, 8);
    return playlists.filter((playlist) =>
      `${playlist.name} ${playlist.description || ""} ${playlist.ownerUsername} ${playlist.tabs.map((tab) => tab.title).join(" ")}`
        .toLowerCase()
        .includes(normalized)
    );
  }, [playlists, query]);

  async function toggleFollow(user: PublicUserProfile) {
    const updated = user.followedByCurrentUser ? await unfollowUser(user.id) : await followUser(user.id);
    setUsers((current) => current.map((item) => (item.id === updated.id ? updated : item)));
  }

  const showTabs = scope === "all" || scope === "tabs";
  const showPlaylists = scope === "all" || scope === "playlists";
  const showUsers = scope === "all" || scope === "users";

  return (
    <main className="page-grid">
      <section className="library-hero">
        <div>
          <p className="eyebrow">Search</p>
          <h2>Find tabs, playlists, and creators.</h2>
          <p>Use one search surface for content and people, then narrow the result type when you know what you want.</p>
        </div>
      </section>

      <section className="library-panel">
        <label className="search-box search-hero-box">
          <Search size={19} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search TaBee"
            autoFocus
          />
        </label>

        <div className="archive-tabs">
          {(["all", "tabs", "playlists", "users"] as SearchScope[]).map((nextScope) => (
            <button key={nextScope} className={scope === nextScope ? "active" : ""} onClick={() => setScope(nextScope)}>
              {nextScope === "tabs" ? <Music size={17} /> : nextScope === "playlists" ? <ListMusic size={17} /> : nextScope === "users" ? <UserRound size={17} /> : <Search size={17} />}
              {labelForScope(nextScope)}
            </button>
          ))}
        </div>

        {error ? <div className="form-error">{error}</div> : null}

        {showTabs ? (
          <SearchSection title="Tabs" count={filteredTabs.length} loading={loading}>
            {filteredTabs.map((tab) => (
              <article className="tab-card" key={tab.id}>
                <Link to={`/tabs/${tab.id}`}>
                  <span className="tab-card-title">{tab.title}</span>
                  <span className="tab-card-meta">
                    {tab.ownerUsername ? `@${tab.ownerUsername} / ` : ""}{tab.fileName} / {tab.instrument}
                  </span>
                </Link>
              </article>
            ))}
          </SearchSection>
        ) : null}

        {showPlaylists ? (
          <SearchSection title="Playlists" count={filteredPlaylists.length} loading={loading}>
            {filteredPlaylists.map((playlist) => (
              <article className="playlist-card" key={playlist.id}>
                <div className="playlist-card-header">
                  <Link to={`/playlists/${playlist.id}`}>
                    <h3>{playlist.name}</h3>
                    <p>by {playlist.ownerUsername} / {playlist.description || `${playlist.tabs.length} tabs`}</p>
                  </Link>
                </div>
              </article>
            ))}
          </SearchSection>
        ) : null}

        {showUsers ? (
          <SearchSection title="Users" count={users.length} loading={searchingUsers}>
            {users.map((user) => (
              <article className="profile-user-row" key={user.id}>
                <Link to={`/users/${user.id}`}>
                  <span>{user.fullName || user.username}</span>
                  <small>@{user.username} / {user.followerCount} followers / {user.followingCount} following</small>
                </Link>
                <button className="btn ghost" onClick={() => toggleFollow(user)}>
                  {user.followedByCurrentUser ? "Following" : "Follow"}
                </button>
              </article>
            ))}
          </SearchSection>
        ) : null}
      </section>
    </main>
  );
}

function SearchSection({
  title,
  count,
  loading,
  children
}: {
  title: string;
  count: number;
  loading: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="search-results-section">
      <div className="section-header">
        <div>
          <h2>{title}</h2>
          <p>{loading ? "Loading..." : `${count} results`}</p>
        </div>
      </div>
      <div className="profile-list">{children}</div>
    </div>
  );
}

function labelForScope(scope: SearchScope) {
  if (scope === "tabs") return "Tabs";
  if (scope === "playlists") return "Playlists";
  if (scope === "users") return "Users";
  return "All";
}

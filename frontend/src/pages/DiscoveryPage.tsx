import { Clock, Compass, ListMusic, Music, Star, UserRound } from "lucide-react";
import type React from "react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { searchUsers } from "../api/authApi";
import {
  favoriteTab,
  listPlaylistArchive,
  listPublicTabs,
  savePlaylist,
  unfavoriteTab,
  unsavePlaylist
} from "../api/tabeeApi";
import { errorMessage } from "../lib/errors";
import type { PublicUserProfile } from "../types/auth";
import type { GeneratedTab, PlaylistResponse } from "../types/tab";

type DiscoveryMode = "forYou" | "tabs" | "playlists" | "creators";
type SortMode = "favorites" | "recent" | "mostTabs";

export function DiscoveryPage() {
  const [mode, setMode] = useState<DiscoveryMode>("forYou");
  const [sortMode, setSortMode] = useState<SortMode>("favorites");
  const [playlists, setPlaylists] = useState<PlaylistResponse[]>([]);
  const [tabs, setTabs] = useState<GeneratedTab[]>([]);
  const [creators, setCreators] = useState<PublicUserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const totalPlaylistTabs = useMemo(
    () => playlists.reduce((count, playlist) => count + playlist.tabs.length, 0),
    [playlists]
  );

  const sortedPlaylists = useMemo(() => {
    return [...playlists].sort((left, right) => {
      if (sortMode === "mostTabs") return right.tabs.length - left.tabs.length;
      if (sortMode === "favorites") return (right.savedCount || 0) - (left.savedCount || 0);
      return new Date(right.createdAt).getTime() - new Date(left.createdAt).getTime();
    });
  }, [playlists, sortMode]);

  const sortedTabs = useMemo(() => {
    return [...tabs].sort((left, right) => {
      if (sortMode === "favorites") {
        return (right.favoriteCount || 0) - (left.favoriteCount || 0);
      }
      return Number(right.id) - Number(left.id);
    });
  }, [sortMode, tabs]);

  const suggestedCreators = useMemo(() => {
    return [...creators].sort((left, right) => right.followerCount - left.followerCount);
  }, [creators]);

  useEffect(() => {
    let active = true;

    async function load() {
      setLoading(true);
      setError("");
      try {
        const [nextPlaylists, nextTabs, nextCreators] = await Promise.all([
          listPlaylistArchive(),
          listPublicTabs(),
          searchUsers("")
        ]);
        if (!active) return;
        setPlaylists(nextPlaylists);
        setTabs(nextTabs);
        setCreators(nextCreators);
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
      <section className="discovery-hero">
        <div>
          <p className="eyebrow">Discovery</p>
          <h2>Browse what the TaBee community is playing.</h2>
          <p>Jump through tabs, playlist collections, and creators without doing an exact search.</p>
        </div>
        <div className="playlist-stat">
          <Compass size={22} />
          <span>{loading ? "Loading..." : `${tabs.length} tabs / ${playlists.length} playlists / ${creators.length} creators`}</span>
        </div>
      </section>

      <section className="library-panel discovery-panel">
        <div className="discovery-mode-tabs">
          {(["forYou", "tabs", "playlists", "creators"] as DiscoveryMode[]).map((nextMode) => (
            <button className={mode === nextMode ? "active" : ""} key={nextMode} onClick={() => setMode(nextMode)}>
              {iconForMode(nextMode)}
              {labelForMode(nextMode)}
            </button>
          ))}
        </div>

        {mode !== "forYou" && mode !== "creators" ? (
          <div className="archive-tabs">
            <button className={sortMode === "favorites" ? "active" : ""} onClick={() => setSortMode("favorites")}>
              <Star size={17} />
              {mode === "playlists" ? "Most saved" : "Most favorited"}
            </button>
            <button className={sortMode === "recent" ? "active" : ""} onClick={() => setSortMode("recent")}>
              <Clock size={17} />
              Most recent
            </button>
            {mode === "playlists" ? (
              <button className={sortMode === "mostTabs" ? "active" : ""} onClick={() => setSortMode("mostTabs")}>
                <ListMusic size={17} />
                Most tabs
              </button>
            ) : null}
          </div>
        ) : null}

        {error ? <div className="form-error">{error}</div> : null}

        {mode === "forYou" ? (
          <>
            <DiscoveryRail title="Tabs picked for you" subtitle={`${sortedTabs.length} public tabs`}>
              {sortedTabs.slice(0, 8).map((tab) => (
                <TabDiscoveryCard tab={tab} key={tab.id} onToggleFavorite={toggleFavorite} />
              ))}
            </DiscoveryRail>
            <DiscoveryRail title="Playlist collections" subtitle={`${sortedPlaylists.length} community playlists`}>
              {sortedPlaylists.slice(0, 8).map((playlist) => (
                <PlaylistDiscoveryCard playlist={playlist} key={playlist.id} onToggleSave={toggleSave} />
              ))}
            </DiscoveryRail>
            <DiscoveryRail title="Creators to follow" subtitle={`${suggestedCreators.length} people`}>
              {suggestedCreators.slice(0, 8).map((creator) => (
                <CreatorDiscoveryCard creator={creator} key={creator.id} />
              ))}
            </DiscoveryRail>
          </>
        ) : null}

        {mode === "tabs" ? (
          <DiscoveryGrid title="Tabs" subtitle={loading ? "Loading tabs..." : `${sortedTabs.length} public tabs`}>
            {sortedTabs.map((tab) => (
              <TabDiscoveryCard tab={tab} key={tab.id} onToggleFavorite={toggleFavorite} />
            ))}
          </DiscoveryGrid>
        ) : null}

        {mode === "playlists" ? (
          <DiscoveryGrid title="Playlists" subtitle={loading ? "Loading playlists..." : `${sortedPlaylists.length} playlists / ${totalPlaylistTabs} tabs`}>
            {sortedPlaylists.map((playlist) => (
              <PlaylistDiscoveryCard playlist={playlist} key={playlist.id} onToggleSave={toggleSave} />
            ))}
          </DiscoveryGrid>
        ) : null}

        {mode === "creators" ? (
          <DiscoveryGrid title="Creators" subtitle={loading ? "Loading creators..." : `${suggestedCreators.length} suggested creators`}>
            {suggestedCreators.map((creator) => (
              <CreatorDiscoveryCard creator={creator} key={creator.id} />
            ))}
          </DiscoveryGrid>
        ) : null}
      </section>
    </main>
  );
}

function DiscoveryRail({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <section className="discovery-rail">
      <div className="section-header">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
      <div className="discovery-strip">{children}</div>
    </section>
  );
}

function DiscoveryGrid({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <section className="discovery-rail">
      <div className="section-header">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
      <div className="discovery-grid">{children}</div>
    </section>
  );
}

function TabDiscoveryCard({ tab, onToggleFavorite }: { tab: GeneratedTab; onToggleFavorite: (tab: GeneratedTab) => void }) {
  return (
    <article className="discovery-card tab-discovery-card">
      <Link to={`/tabs/${tab.id}`}>
        <div className="discovery-art tab-art">
          <Music size={28} />
        </div>
        <strong>{tab.title}</strong>
        <span>{tab.ownerUsername ? `@${tab.ownerUsername}` : tab.fileName}</span>
        <span className="discovery-card-stat">
          <Star size={14} />
          {formatFavoriteCount(tab.favoriteCount || 0)}
        </span>
      </Link>
      {!tab.createdByCurrentUser ? (
        <button
          className={`icon-btn subtle floating-action${tab.favoritedByCurrentUser ? " active" : ""}`}
          title={tab.favoritedByCurrentUser ? "Favorited" : "Favorite tab"}
          onClick={() => onToggleFavorite(tab)}
        >
          <Star size={17} />
        </button>
      ) : null}
    </article>
  );
}

function PlaylistDiscoveryCard({ playlist, onToggleSave }: { playlist: PlaylistResponse; onToggleSave: (playlist: PlaylistResponse) => void }) {
  return (
    <article className="discovery-card playlist-discovery-card">
      <Link to={`/playlists/${playlist.id}`}>
        <div className="discovery-art playlist-art">
          <ListMusic size={28} />
        </div>
        <strong>{playlist.name}</strong>
        <span>by @{playlist.ownerUsername} / {playlist.tabs.length} tabs</span>
        <span>Created {formatDiscoveryDate(playlist.createdAt)}</span>
        <span className="discovery-card-stat">
          <Star size={14} />
          {formatSaveCount(playlist.savedCount || 0)}
        </span>
      </Link>
      {!playlist.createdByCurrentUser ? (
        <button
          className={`icon-btn subtle floating-action${playlist.savedByCurrentUser ? " active" : ""}`}
          title={playlist.savedByCurrentUser ? "Saved" : "Save playlist"}
          onClick={() => onToggleSave(playlist)}
        >
          <Star size={17} />
        </button>
      ) : null}
    </article>
  );
}

function CreatorDiscoveryCard({ creator }: { creator: PublicUserProfile }) {
  return (
    <article className="discovery-card creator-discovery-card">
      <Link to={`/users/${creator.id}`}>
        <div className="discovery-art creator-art">
          <UserRound size={28} />
        </div>
        <strong>{creator.fullName || creator.username}</strong>
        <span>@{creator.username} / {creator.followerCount} followers</span>
      </Link>
    </article>
  );
}

function iconForMode(mode: DiscoveryMode) {
  if (mode === "tabs") return <Music size={17} />;
  if (mode === "playlists") return <ListMusic size={17} />;
  if (mode === "creators") return <UserRound size={17} />;
  return <Compass size={17} />;
}

function labelForMode(mode: DiscoveryMode) {
  if (mode === "tabs") return "Tabs";
  if (mode === "playlists") return "Playlists";
  if (mode === "creators") return "Creators";
  return "For You";
}

function formatDiscoveryDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function formatFavoriteCount(count: number) {
  return `${count} ${count === 1 ? "Favorite" : "Favorites"}`;
}

function formatSaveCount(count: number) {
  return `${count} ${count === 1 ? "Save" : "Saves"}`;
}

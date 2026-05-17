import { ListMusic, Music, UserRound } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { followUser, getPublicUser, unfollowUser } from "../api/authApi";
import { listPlaylistArchiveByUser, listPublicTabsByUser } from "../api/tabeeApi";
import { errorMessage } from "../lib/errors";
import type { PublicUserProfile } from "../types/auth";
import type { GeneratedTab, PlaylistResponse } from "../types/tab";

export function PublicUserPage() {
  const { userId } = useParams();
  const [profile, setProfile] = useState<PublicUserProfile | null>(null);
  const [tabs, setTabs] = useState<GeneratedTab[]>([]);
  const [playlists, setPlaylists] = useState<PlaylistResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadUser() {
      if (!userId) return;
      setLoading(true);
      setError("");
      try {
        const [nextProfile, nextTabs, nextPlaylists] = await Promise.all([
          getPublicUser(userId),
          listPublicTabsByUser(userId),
          listPlaylistArchiveByUser(userId)
        ]);
        if (!active) return;
        setProfile(nextProfile);
        setTabs(nextTabs);
        setPlaylists(nextPlaylists);
      } catch (caught) {
        if (active) {
          setError(errorMessage(caught, "Could not load user."));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadUser();
    return () => {
      active = false;
    };
  }, [userId]);

  async function toggleFollow() {
    if (!profile) return;
    const updated = profile.followedByCurrentUser ? await unfollowUser(profile.id) : await followUser(profile.id);
    setProfile(updated);
  }

  if (!profile && loading) {
    return (
      <main className="page-grid">
        <div className="empty-panel">
          <h2>Loading profile</h2>
          <p>Fetching creator activity.</p>
        </div>
      </main>
    );
  }

  if (!profile) {
    return (
      <main className="page-grid">
        <div className="empty-panel">
          <h2>User not found</h2>
          <p>{error || "This profile is unavailable."}</p>
        </div>
      </main>
    );
  }

  return (
    <main className="profile-page">
      <section className="profile-hero">
        <div className="profile-avatar">
          <UserRound size={28} />
        </div>
        <div>
          <p className="eyebrow">Creator</p>
          <h2>{profile.fullName || profile.username}</h2>
          <p>@{profile.username}</p>
          <div className="social-counts">
            <Link to={`/users/${profile.id}/followers`}>
              <strong>{profile.followerCount}</strong>
              <span>Followers</span>
            </Link>
            <Link to={`/users/${profile.id}/following`}>
              <strong>{profile.followingCount}</strong>
              <span>Following</span>
            </Link>
          </div>
        </div>
        <button className={`btn ${profile.followedByCurrentUser ? "ghost" : "primary"}`} onClick={toggleFollow}>
          {profile.followedByCurrentUser ? "Following" : "Follow"}
        </button>
      </section>

      {error ? <div className="form-error">{error}</div> : null}

      <section className="profile-grid">
        <div className="profile-panel">
          <div className="section-header">
            <div>
              <h2>Tabs</h2>
              <p>{tabs.length} public tabs</p>
            </div>
            <Music size={20} />
          </div>
          <div className="profile-list">
            {tabs.map((tab) => (
              <Link className="profile-tab-row" to={`/tabs/${tab.id}`} key={tab.id}>
                <span>{tab.title}</span>
                <small>{tab.fileName} / {tab.instrument}</small>
              </Link>
            ))}
          </div>
        </div>

        <div className="profile-panel">
          <div className="section-header">
            <div>
              <h2>Playlists</h2>
              <p>{playlists.length} created playlists</p>
            </div>
            <ListMusic size={20} />
          </div>
          <div className="profile-list">
            {playlists.map((playlist) => (
              <Link className="profile-playlist-row" to={`/playlists/${playlist.id}`} key={playlist.id}>
                <div>
                  <span>{playlist.name}</span>
                  <small>{playlist.description || `${playlist.tabs.length} tabs`}</small>
                </div>
                <strong>{playlist.tabs.length}</strong>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

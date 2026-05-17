import { ListMusic, Music } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { followUser, getPublicUser, unfollowUser } from "../api/authApi";
import { listPlaylistArchiveByUser, listPublicTabsByUser } from "../api/tabeeApi";
import { Avatar } from "../components/Avatar";
import { PlaylistCover } from "../components/PlaylistCover";
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
          <p>Preparing this user profile.</p>
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
      <section className="profile-hero public-profile-hero">
        <div className="public-profile-media">
          <Avatar src={profile.profileImageUrl} label={profile.username} size="lg" />
          <button className={`btn ${profile.followedByCurrentUser ? "ghost" : "primary"}`} onClick={toggleFollow}>
            {profile.followedByCurrentUser ? "Following" : "Follow"}
          </button>
        </div>
        <div className="public-profile-summary">
          <h2>{profile.fullName || profile.username}</h2>
          <p className="public-profile-username">@{profile.username}</p>
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
        <div className="public-profile-facts">
          <span>
            <strong>{tabs.length}</strong>
            <small>Tabs</small>
          </span>
          <span>
            <strong>{playlists.length}</strong>
            <small>Playlists</small>
          </span>
        </div>
      </section>

      {error ? <div className="form-error">{error}</div> : null}

      <section className="profile-grid equal-profile-grid">
        <div className="profile-panel public-profile-panel">
          <div className="section-header">
            <div>
              <h2>Tabs</h2>
              <p>{tabs.length} public tabs</p>
            </div>
            <Music size={20} />
          </div>
          <div className="profile-list scroll-list">
            {tabs.map((tab) => (
              <Link className="profile-tab-row" to={`/tabs/${tab.id}`} key={tab.id}>
                <span>{tab.title}</span>
                <small>{tab.fileName} / {tab.instrument}</small>
              </Link>
            ))}
            {!tabs.length ? (
              <div className="empty-list-message">
                <h3>{profile.username} has not created any tabs yet</h3>
                <p>Generated tabs will appear here when they are shared.</p>
              </div>
            ) : null}
          </div>
        </div>

        <div className="profile-panel public-profile-panel">
          <div className="section-header">
            <div>
              <h2>Playlists</h2>
              <p>{playlists.length} created playlists</p>
            </div>
            <ListMusic size={20} />
          </div>
          <div className="profile-list scroll-list">
            {playlists.map((playlist) => (
              <Link className="profile-playlist-row" to={`/playlists/${playlist.id}`} key={playlist.id}>
                <PlaylistCover src={playlist.coverImageUrl} title={playlist.name} size="sm" />
                <div>
                  <span>{playlist.name}</span>
                  <small>{playlist.description || `${playlist.tabs.length} tabs`}</small>
                </div>
                <strong>{playlist.tabs.length}</strong>
              </Link>
            ))}
            {!playlists.length ? (
              <div className="empty-list-message">
                <h3>{profile.username} has not created a playlist yet</h3>
                <p>Playlists they publish will show up in this panel.</p>
              </div>
            ) : null}
          </div>
        </div>
      </section>
    </main>
  );
}

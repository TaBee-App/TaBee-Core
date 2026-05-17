import { UserRound } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  followUser,
  getPublicUser,
  getUserFollowers,
  getUserFollowing,
  removeFollower,
  unfollowUser
} from "../api/authApi";
import { getCurrentUser } from "../api/authSession";
import { errorMessage } from "../lib/errors";
import type { PublicUserProfile } from "../types/auth";

export function UserConnectionsPage() {
  const { userId, kind } = useParams();
  const currentUser = getCurrentUser();
  const [profile, setProfile] = useState<PublicUserProfile | null>(null);
  const [users, setUsers] = useState<PublicUserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const isFollowers = kind === "followers";
  const isOwnConnections = Boolean(currentUser?.id && userId && String(currentUser.id) === userId);

  useEffect(() => {
    let active = true;

    async function loadConnections() {
      if (!userId) return;
      setLoading(true);
      setError("");
      try {
        const [nextProfile, nextUsers] = await Promise.all([
          getPublicUser(userId),
          isFollowers ? getUserFollowers(userId) : getUserFollowing(userId)
        ]);
        if (!active) return;
        setProfile(nextProfile);
        setUsers(nextUsers);
      } catch (caught) {
        if (active) {
          setError(errorMessage(caught, "Could not load users."));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadConnections();
    return () => {
      active = false;
    };
  }, [isFollowers, userId]);

  async function toggleFollow(user: PublicUserProfile) {
    const updated = user.followedByCurrentUser ? await unfollowUser(user.id) : await followUser(user.id);
    setUsers((current) => current.map((item) => (item.id === updated.id ? updated : item)));
  }

  async function removeUserFollower(user: PublicUserProfile) {
    setError("");
    try {
      await removeFollower(user.id);
      setUsers((current) => current.filter((item) => item.id !== user.id));
      setProfile((current) =>
        current
          ? {
              ...current,
              followerCount: Math.max(0, current.followerCount - 1)
            }
          : current
      );
    } catch (caught) {
      setError(errorMessage(caught, "Could not remove follower."));
    }
  }

  return (
    <main className="page-grid">
      <section className="library-hero">
        <div>
          <p className="eyebrow">{isFollowers ? "Followers" : "Following"}</p>
          <h2>{profile ? `${profile.fullName || profile.username}` : "User connections"}</h2>
          <p>
            {loading ? "Loading..." : `${users.length} ${isFollowers ? "followers" : "following"}`}
            {profile ? ` for @${profile.username}` : ""}
          </p>
        </div>
      </section>

      {error ? <div className="form-error">{error}</div> : null}

      <section className="library-panel">
        <div className="profile-list">
          {users.map((user) => (
            <article className="profile-user-row" key={user.id}>
              <Link to={currentUser?.id === user.id ? "/profile" : `/users/${user.id}`}>
                <span>{user.fullName || user.username}</span>
                <small>
                  @{user.username} / {user.followerCount} followers / {user.followingCount} following
                  {currentUser?.id === user.id ? " / you" : ""}
                </small>
              </Link>
              {currentUser?.id !== user.id ? (
                <div className="profile-row-actions">
                  {isFollowers && isOwnConnections ? (
                    <button className="btn ghost danger" onClick={() => removeUserFollower(user)}>
                      Remove
                    </button>
                  ) : null}
                  <button className="btn ghost" onClick={() => toggleFollow(user)}>
                    {user.followedByCurrentUser ? "Following" : "Follow"}
                  </button>
                </div>
              ) : null}
            </article>
          ))}
        </div>

        {!loading && !users.length ? (
          <div className="empty-panel compact">
            <UserRound size={24} />
            <h3>No users yet</h3>
            <p>This list is empty.</p>
          </div>
        ) : null}
      </section>
    </main>
  );
}

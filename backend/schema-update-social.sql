BEGIN;

-- Saved playlists: separates playlists created by the current user from
-- playlists created by other users that the current user wants to keep.
CREATE TABLE IF NOT EXISTS saved_playlists (
    user_id BIGINT NOT NULL,
    playlist_id BIGINT NOT NULL,
    saved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_saved_playlists
        PRIMARY KEY (user_id, playlist_id),

    CONSTRAINT fk_saved_playlists_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_saved_playlists_playlist
        FOREIGN KEY (playlist_id)
        REFERENCES user_playlists(playlist_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_saved_playlists_user_saved_at
    ON saved_playlists(user_id, saved_at DESC);

CREATE INDEX IF NOT EXISTS idx_saved_playlists_playlist_id
    ON saved_playlists(playlist_id);

-- User follows: lets discovery later distinguish all public content from
-- content created by people the current user follows.
CREATE TABLE IF NOT EXISTS user_follows (
    follower_user_id BIGINT NOT NULL,
    followed_user_id BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_user_follows
        PRIMARY KEY (follower_user_id, followed_user_id),

    CONSTRAINT fk_user_follows_follower
        FOREIGN KEY (follower_user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_user_follows_followed
        FOREIGN KEY (followed_user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_user_follows_not_self
        CHECK (follower_user_id <> followed_user_id)
);

CREATE INDEX IF NOT EXISTS idx_user_follows_follower_created_at
    ON user_follows(follower_user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_follows_followed_created_at
    ON user_follows(followed_user_id, created_at DESC);

-- Favorite tabs: separates tabs created by the current user from tabs
-- created by other users that the current user wants to keep.
CREATE TABLE IF NOT EXISTS favorite_tabs (
    user_id BIGINT NOT NULL,
    tab_id BIGINT NOT NULL,
    favorited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_favorite_tabs
        PRIMARY KEY (user_id, tab_id),

    CONSTRAINT fk_favorite_tabs_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_favorite_tabs_tab
        FOREIGN KEY (tab_id)
        REFERENCES tabs(tab_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_favorite_tabs_user_favorited_at
    ON favorite_tabs(user_id, favorited_at DESC);

CREATE INDEX IF NOT EXISTS idx_favorite_tabs_tab_id
    ON favorite_tabs(tab_id);

COMMIT;

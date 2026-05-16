package com.tabee.backend.playlist;

import java.io.Serializable;
import java.util.Objects;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;

@Embeddable
public class SavedPlaylistId implements Serializable {
    @Column(name = "user_id")
    private Long userId;

    @Column(name = "playlist_id")
    private Long playlistId;

    public SavedPlaylistId() {
    }

    public SavedPlaylistId(Long userId, Long playlistId) {
        this.userId = userId;
        this.playlistId = playlistId;
    }

    public Long getUserId() {
        return userId;
    }

    public Long getPlaylistId() {
        return playlistId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof SavedPlaylistId that)) {
            return false;
        }
        return Objects.equals(userId, that.userId) && Objects.equals(playlistId, that.playlistId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(userId, playlistId);
    }
}

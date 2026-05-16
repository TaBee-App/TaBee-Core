package com.tabee.backend.playlist;

import java.time.OffsetDateTime;

import com.tabee.backend.user.User;

import jakarta.persistence.Column;
import jakarta.persistence.EmbeddedId;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.MapsId;
import jakarta.persistence.Table;

@Entity
@Table(name = "saved_playlists")
public class SavedPlaylist {
    @EmbeddedId
    private SavedPlaylistId id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @MapsId("userId")
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @MapsId("playlistId")
    @JoinColumn(name = "playlist_id", nullable = false)
    private UserPlaylist playlist;

    @Column(name = "saved_at", nullable = false)
    private OffsetDateTime savedAt = OffsetDateTime.now();

    public SavedPlaylistId getId() {
        return id;
    }

    public void setId(SavedPlaylistId id) {
        this.id = id;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public UserPlaylist getPlaylist() {
        return playlist;
    }

    public void setPlaylist(UserPlaylist playlist) {
        this.playlist = playlist;
    }

    public OffsetDateTime getSavedAt() {
        return savedAt;
    }
}

package com.tabee.backend.playlist;

import java.io.Serializable;
import java.util.Objects;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;

@Embeddable
public class PlaylistTabId implements Serializable {
    @Column(name = "playlist_id")
    private Long playlistId;

    @Column(name = "tab_id")
    private Long tabId;

    public PlaylistTabId() {
    }

    public PlaylistTabId(Long playlistId, Long tabId) {
        this.playlistId = playlistId;
        this.tabId = tabId;
    }

    public Long getPlaylistId() {
        return playlistId;
    }

    public Long getTabId() {
        return tabId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof PlaylistTabId that)) {
            return false;
        }
        return Objects.equals(playlistId, that.playlistId) && Objects.equals(tabId, that.tabId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(playlistId, tabId);
    }
}

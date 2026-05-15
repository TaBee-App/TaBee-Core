package com.tabee.backend.playlist;

import java.time.OffsetDateTime;

import com.tabee.backend.tab.Tab;

import jakarta.persistence.Column;
import jakarta.persistence.EmbeddedId;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.MapsId;
import jakarta.persistence.Table;

@Entity
@Table(name = "playlist_tabs")
public class PlaylistTab {

    @EmbeddedId
    private PlaylistTabId id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @MapsId("playlistId")
    @JoinColumn(name = "playlist_id", nullable = false)
    private UserPlaylist playlist;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @MapsId("tabId")
    @JoinColumn(name = "tab_id", nullable = false)
    private Tab tab;

    @Column(name = "added_at", nullable = false)
    private OffsetDateTime addedAt = OffsetDateTime.now();

    public PlaylistTabId getId() {
        return id;
    }

    public void setId(PlaylistTabId id) {
        this.id = id;
    }

    public UserPlaylist getPlaylist() {
        return playlist;
    }

    public void setPlaylist(UserPlaylist playlist) {
        this.playlist = playlist;
    }

    public Tab getTab() {
        return tab;
    }

    public void setTab(Tab tab) {
        this.tab = tab;
    }

    public OffsetDateTime getAddedAt() {
        return addedAt;
    }
}

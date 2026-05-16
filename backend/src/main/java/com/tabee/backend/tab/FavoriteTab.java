package com.tabee.backend.tab;

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
@Table(name = "favorite_tabs")
public class FavoriteTab {
    @EmbeddedId
    private FavoriteTabId id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @MapsId("userId")
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @MapsId("tabId")
    @JoinColumn(name = "tab_id", nullable = false)
    private Tab tab;

    @Column(name = "favorited_at", nullable = false)
    private OffsetDateTime favoritedAt = OffsetDateTime.now();

    public FavoriteTabId getId() {
        return id;
    }

    public void setId(FavoriteTabId id) {
        this.id = id;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public Tab getTab() {
        return tab;
    }

    public void setTab(Tab tab) {
        this.tab = tab;
    }

    public OffsetDateTime getFavoritedAt() {
        return favoritedAt;
    }
}

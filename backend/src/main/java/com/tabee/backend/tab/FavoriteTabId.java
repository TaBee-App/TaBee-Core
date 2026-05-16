package com.tabee.backend.tab;

import java.io.Serializable;
import java.util.Objects;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;

@Embeddable
public class FavoriteTabId implements Serializable {
    @Column(name = "user_id")
    private Long userId;

    @Column(name = "tab_id")
    private Long tabId;

    public FavoriteTabId() {
    }

    public FavoriteTabId(Long userId, Long tabId) {
        this.userId = userId;
        this.tabId = tabId;
    }

    public Long getUserId() {
        return userId;
    }

    public Long getTabId() {
        return tabId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof FavoriteTabId that)) {
            return false;
        }
        return Objects.equals(userId, that.userId) && Objects.equals(tabId, that.tabId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(userId, tabId);
    }
}

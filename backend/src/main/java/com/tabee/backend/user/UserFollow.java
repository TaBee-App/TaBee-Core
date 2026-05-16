package com.tabee.backend.user;

import java.time.OffsetDateTime;

import jakarta.persistence.Column;
import jakarta.persistence.EmbeddedId;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.MapsId;
import jakarta.persistence.Table;

@Entity
@Table(name = "user_follows")
public class UserFollow {
    @EmbeddedId
    private UserFollowId id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @MapsId("followerUserId")
    @JoinColumn(name = "follower_user_id", nullable = false)
    private User follower;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @MapsId("followedUserId")
    @JoinColumn(name = "followed_user_id", nullable = false)
    private User followed;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt = OffsetDateTime.now();

    public UserFollowId getId() {
        return id;
    }

    public void setId(UserFollowId id) {
        this.id = id;
    }

    public User getFollower() {
        return follower;
    }

    public void setFollower(User follower) {
        this.follower = follower;
    }

    public User getFollowed() {
        return followed;
    }

    public void setFollowed(User followed) {
        this.followed = followed;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }
}

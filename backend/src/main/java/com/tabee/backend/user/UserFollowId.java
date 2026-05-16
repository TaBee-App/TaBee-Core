package com.tabee.backend.user;

import java.io.Serializable;
import java.util.Objects;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;

@Embeddable
public class UserFollowId implements Serializable {
    @Column(name = "follower_user_id")
    private Long followerUserId;

    @Column(name = "followed_user_id")
    private Long followedUserId;

    public UserFollowId() {
    }

    public UserFollowId(Long followerUserId, Long followedUserId) {
        this.followerUserId = followerUserId;
        this.followedUserId = followedUserId;
    }

    public Long getFollowerUserId() {
        return followerUserId;
    }

    public Long getFollowedUserId() {
        return followedUserId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof UserFollowId that)) {
            return false;
        }
        return Objects.equals(followerUserId, that.followerUserId)
                && Objects.equals(followedUserId, that.followedUserId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(followerUserId, followedUserId);
    }
}

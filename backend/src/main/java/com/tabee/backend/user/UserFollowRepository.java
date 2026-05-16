package com.tabee.backend.user;

import java.util.List;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserFollowRepository extends JpaRepository<UserFollow, UserFollowId> {
    @EntityGraph(attributePaths = {"followed"})
    List<UserFollow> findByFollower_IdOrderByCreatedAtDesc(Long followerUserId);

    @EntityGraph(attributePaths = {"follower"})
    List<UserFollow> findByFollowed_IdOrderByCreatedAtDesc(Long followedUserId);

    boolean existsByFollower_IdAndFollowed_Id(Long followerUserId, Long followedUserId);

    long countByFollower_Id(Long followerUserId);

    long countByFollowed_Id(Long followedUserId);
}

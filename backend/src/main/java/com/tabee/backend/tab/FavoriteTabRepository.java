package com.tabee.backend.tab;

import java.util.List;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface FavoriteTabRepository extends JpaRepository<FavoriteTab, FavoriteTabId> {
    @EntityGraph(attributePaths = {"tab", "tab.owner", "tab.tabData"})
    List<FavoriteTab> findByUser_IdOrderByFavoritedAtDesc(Long userId);

    boolean existsByUser_IdAndTab_Id(Long userId, Long tabId);

    @Query("select favorite.id.tabId from FavoriteTab favorite where favorite.user.id = :userId")
    List<Long> findTabIdsByUserId(@Param("userId") Long userId);
}

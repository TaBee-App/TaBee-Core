package com.tabee.backend.playlist;

import java.util.Optional;

import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface PlaylistTabRepository extends JpaRepository<PlaylistTab, PlaylistTabId> {
    Optional<PlaylistTab> findByIdPlaylistIdAndIdTabId(Long playlistId, Long tabId);

    @Modifying
    @Query(value = """
            INSERT INTO playlist_tabs (playlist_id, tab_id)
            VALUES (:playlistId, :tabId)
            ON CONFLICT (playlist_id, tab_id) DO NOTHING
            """, nativeQuery = true)
    int insertIgnoreConflict(@Param("playlistId") Long playlistId, @Param("tabId") Long tabId);

    @Modifying
    @Query(value = """
            DELETE FROM playlist_tabs
            WHERE playlist_id = :playlistId AND tab_id = :tabId
            """, nativeQuery = true)
    int deleteByPlaylistIdAndTabId(@Param("playlistId") Long playlistId, @Param("tabId") Long tabId);
}

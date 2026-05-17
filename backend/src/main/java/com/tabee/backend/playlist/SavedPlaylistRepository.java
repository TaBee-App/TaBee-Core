package com.tabee.backend.playlist;

import java.util.List;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

public interface SavedPlaylistRepository extends JpaRepository<SavedPlaylist, SavedPlaylistId> {
    @EntityGraph(attributePaths = {"playlist", "playlist.owner", "playlist.playlistTabs", "playlist.playlistTabs.tab", "playlist.playlistTabs.tab.owner"})
    List<SavedPlaylist> findByUser_IdOrderBySavedAtDesc(Long userId);

    boolean existsByUser_IdAndPlaylist_Id(Long userId, Long playlistId);

    long countByPlaylist_Id(Long playlistId);
}

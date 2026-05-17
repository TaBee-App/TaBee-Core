package com.tabee.backend.playlist;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserPlaylistRepository extends JpaRepository<UserPlaylist, Long> {
    @EntityGraph(attributePaths = {"owner", "playlistTabs", "playlistTabs.tab", "playlistTabs.tab.owner"})
    List<UserPlaylist> findByOwnerIdOrderByCreatedAtDesc(Long ownerId);

    @EntityGraph(attributePaths = {"owner", "playlistTabs", "playlistTabs.tab", "playlistTabs.tab.owner"})
    List<UserPlaylist> findAllByOrderByCreatedAtDesc();

    @EntityGraph(attributePaths = {"owner", "playlistTabs", "playlistTabs.tab", "playlistTabs.tab.owner"})
    List<UserPlaylist> findTop10ByOrderByCreatedAtDesc();

    @Override
    @EntityGraph(attributePaths = {"owner", "playlistTabs", "playlistTabs.tab", "playlistTabs.tab.owner"})
    Optional<UserPlaylist> findById(Long id);

    @EntityGraph(attributePaths = {"owner", "playlistTabs", "playlistTabs.tab", "playlistTabs.tab.owner"})
    Optional<UserPlaylist> findByIdAndOwnerId(Long id, Long ownerId);
}

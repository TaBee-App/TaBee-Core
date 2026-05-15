package com.tabee.backend.playlist;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserPlaylistRepository extends JpaRepository<UserPlaylist, Long> {
    @EntityGraph(attributePaths = {"owner", "playlistTabs", "playlistTabs.tab"})
    List<UserPlaylist> findByOwnerIdOrderByCreatedAtDesc(Long ownerId);

    @Override
    @EntityGraph(attributePaths = {"owner", "playlistTabs", "playlistTabs.tab"})
    Optional<UserPlaylist> findById(Long id);
}

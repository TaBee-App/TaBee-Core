package com.tabee.backend.audio;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AudioFileRepository extends JpaRepository<AudioFile, Long> {
    @EntityGraph(attributePaths = "owner")
    List<AudioFile> findByOwnerIdOrderByUploadedAtDesc(Long ownerId);

    @Override
    @EntityGraph(attributePaths = "owner")
    Optional<AudioFile> findById(Long id);
}

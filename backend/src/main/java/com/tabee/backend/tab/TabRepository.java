package com.tabee.backend.tab;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TabRepository extends JpaRepository<Tab, Long> {
    @EntityGraph(attributePaths = {"owner", "tabData"})
    List<Tab> findByOwnerIdOrderByCreatedAtDesc(Long ownerId);

    @EntityGraph(attributePaths = {"owner", "tabData"})
    List<Tab> findAllByOrderByCreatedAtDesc();

    @Override
    @EntityGraph(attributePaths = {"owner", "tabData"})
    Optional<Tab> findById(Long id);

    @EntityGraph(attributePaths = {"owner", "tabData"})
    Optional<Tab> findByIdAndOwnerId(Long id, Long ownerId);
}

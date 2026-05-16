package com.tabee.backend.tab;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TabRepository extends JpaRepository<Tab, Long> {
    @EntityGraph(attributePaths = {"owner", "tabData"})
    List<Tab> findByOwnerIdOrderByCreatedAtDesc(Long ownerId);

    @Override
    @EntityGraph(attributePaths = {"owner", "tabData"})
    Optional<Tab> findById(Long id);
}

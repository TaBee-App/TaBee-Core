package com.tabee.backend.tab;

import java.time.OffsetDateTime;

import com.fasterxml.jackson.databind.JsonNode;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

public final class TabDtos {
    private TabDtos() {
    }

    public record TabRequest(
            @NotBlank String title,
            String artist,
            @Size(max = 50) String tuning,
            Integer estimatedTempo,
            @NotNull JsonNode jsonData
    ) {
    }

    public record TabUpdateRequest(
            String title,
            String artist,
            @Size(max = 50) String tuning,
            Integer estimatedTempo,
            JsonNode jsonData
    ) {
    }

    public record TabResponse(
            Long id,
            Long ownerUserId,
            String ownerUsername,
            String ownerProfileImageUrl,
            Long tabDataId,
            String title,
            String artist,
            String tuning,
            Integer estimatedTempo,
            OffsetDateTime createdAt,
            OffsetDateTime updatedAt,
            boolean createdByCurrentUser,
            boolean favoritedByCurrentUser,
            long favoriteCount,
            JsonNode jsonData
    ) {
        public static TabResponse from(Tab tab) {
            return from(tab, null, false, 0, null);
        }

        public static TabResponse from(Tab tab, Long currentUserId, boolean favoritedByCurrentUser) {
            return from(tab, currentUserId, favoritedByCurrentUser, 0, null);
        }

        public static TabResponse from(Tab tab, Long currentUserId, boolean favoritedByCurrentUser, long favoriteCount) {
            return from(tab, currentUserId, favoritedByCurrentUser, favoriteCount, null);
        }

        public static TabResponse from(Tab tab, Long currentUserId, boolean favoritedByCurrentUser, long favoriteCount,
                                       String ownerProfileImageUrl) {
            return new TabResponse(
                    tab.getId(),
                    tab.getOwner().getId(),
                    tab.getOwner().getUsername(),
                    ownerProfileImageUrl,
                    tab.getTabData().getId(),
                    tab.getTitle(),
                    tab.getArtist(),
                    tab.getTabData().getTuning(),
                    tab.getTabData().getEstimatedTempo(),
                    tab.getCreatedAt(),
                    tab.getUpdatedAt(),
                    currentUserId != null && tab.getOwner().getId().equals(currentUserId),
                    favoritedByCurrentUser,
                    favoriteCount,
                    tab.getTabData().getJsonData()
            );
        }
    }
}

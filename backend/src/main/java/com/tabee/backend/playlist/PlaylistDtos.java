package com.tabee.backend.playlist;

import java.time.OffsetDateTime;
import java.util.List;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

public final class PlaylistDtos {
    private PlaylistDtos() {
    }

    public record PlaylistRequest(
            @NotBlank @Size(max = 100) String name,
            String description
    ) {
    }

    public record AddTabRequest(
            @NotNull Long tabId
    ) {
    }

    public record PlaylistTabResponse(
            Long tabId,
            Long ownerUserId,
            String title,
            String artist,
            OffsetDateTime addedAt
    ) {
        public static PlaylistTabResponse from(PlaylistTab playlistTab) {
            return new PlaylistTabResponse(
                    playlistTab.getTab().getId(),
                    playlistTab.getTab().getOwner().getId(),
                    playlistTab.getTab().getTitle(),
                    playlistTab.getTab().getArtist(),
                    playlistTab.getAddedAt()
            );
        }
    }

    public record PlaylistResponse(
            Long id,
            Long ownerUserId,
            String ownerUsername,
            String name,
            String description,
            OffsetDateTime createdAt,
            boolean createdByCurrentUser,
            boolean savedByCurrentUser,
            List<PlaylistTabResponse> tabs
    ) {
        public static PlaylistResponse from(UserPlaylist playlist) {
            return from(playlist, null, false);
        }

        public static PlaylistResponse from(UserPlaylist playlist, Long currentUserId, boolean savedByCurrentUser) {
            boolean createdByCurrentUser = currentUserId != null && playlist.getOwner().getId().equals(currentUserId);
            return new PlaylistResponse(
                    playlist.getId(),
                    playlist.getOwner().getId(),
                    playlist.getOwner().getUsername(),
                    playlist.getName(),
                    playlist.getDescription(),
                    playlist.getCreatedAt(),
                    createdByCurrentUser,
                    savedByCurrentUser,
                    playlist.getPlaylistTabs().stream().map(PlaylistTabResponse::from).toList()
            );
        }
    }
}

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
            String title,
            String artist,
            OffsetDateTime addedAt
    ) {
        public static PlaylistTabResponse from(PlaylistTab playlistTab) {
            return new PlaylistTabResponse(
                    playlistTab.getTab().getId(),
                    playlistTab.getTab().getTitle(),
                    playlistTab.getTab().getArtist(),
                    playlistTab.getAddedAt()
            );
        }
    }

    public record PlaylistResponse(
            Long id,
            Long ownerUserId,
            String name,
            String description,
            OffsetDateTime createdAt,
            List<PlaylistTabResponse> tabs
    ) {
        public static PlaylistResponse from(UserPlaylist playlist) {
            return new PlaylistResponse(
                    playlist.getId(),
                    playlist.getOwner().getId(),
                    playlist.getName(),
                    playlist.getDescription(),
                    playlist.getCreatedAt(),
                    playlist.getPlaylistTabs().stream().map(PlaylistTabResponse::from).toList()
            );
        }
    }
}

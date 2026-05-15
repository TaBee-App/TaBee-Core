package com.tabee.backend.tab;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.PositiveOrZero;
import jakarta.validation.constraints.Size;

public final class TabDtos {
    private TabDtos() {
    }

    public record NoteEventRequest(
            @NotNull @PositiveOrZero BigDecimal time,
            BigDecimal frequency,
            BigDecimal confidence,
            @Size(max = 10) String noteName,
            Integer midiNumber,
            @PositiveOrZero Integer fret,
            @Positive Integer stringNumber
    ) {
    }

    public record NoteEventResponse(
            Long id,
            BigDecimal time,
            BigDecimal frequency,
            BigDecimal confidence,
            String noteName,
            Integer midiNumber,
            Integer fret,
            Integer stringNumber
    ) {
        public static NoteEventResponse from(NoteEvent noteEvent) {
            return new NoteEventResponse(
                    noteEvent.getId(),
                    noteEvent.getTime(),
                    noteEvent.getFrequency(),
                    noteEvent.getConfidence(),
                    noteEvent.getNoteName(),
                    noteEvent.getMidiNumber(),
                    noteEvent.getFret(),
                    noteEvent.getStringNumber()
            );
        }
    }

    public record TabRequest(
            @NotNull Long sourceAudioId,
            @NotBlank String title,
            String artist,
            @Size(max = 50) String tuning,
            Integer estimatedTempo,
            @Valid List<NoteEventRequest> notes
    ) {
    }

    public record TabUpdateRequest(
            String title,
            String artist,
            @Size(max = 50) String tuning,
            Integer estimatedTempo,
            @Valid List<NoteEventRequest> notes
    ) {
    }

    public record TabResponse(
            Long id,
            Long ownerUserId,
            Long sourceAudioId,
            Long tabDataId,
            String title,
            String artist,
            String tuning,
            Integer estimatedTempo,
            OffsetDateTime createdAt,
            OffsetDateTime updatedAt,
            List<NoteEventResponse> notes
    ) {
        public static TabResponse from(Tab tab) {
            return new TabResponse(
                    tab.getId(),
                    tab.getOwner().getId(),
                    tab.getSourceAudio().getId(),
                    tab.getTabData().getId(),
                    tab.getTitle(),
                    tab.getArtist(),
                    tab.getTabData().getTuning(),
                    tab.getTabData().getEstimatedTempo(),
                    tab.getCreatedAt(),
                    tab.getUpdatedAt(),
                    tab.getTabData().getNoteEvents().stream().map(NoteEventResponse::from).toList()
            );
        }
    }
}

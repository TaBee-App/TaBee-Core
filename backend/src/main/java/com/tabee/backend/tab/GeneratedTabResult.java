package com.tabee.backend.tab;

import java.math.BigDecimal;
import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;

public record GeneratedTabResult(
        String sourceAudio,
        String instrument,
        String tuning,
        Integer estimatedTempo,
        Integer beatsPerBar,
        JsonNode algorithm,
        Summary summary,
        List<GeneratedNoteEvent> noteEvents
) {
    public record Summary(
            Integer detectedOnsets,
            Integer detectedNotes,
            Integer playableNotes
    ) {
    }

    public record GeneratedNoteEvent(
            Boolean isRest,
            BigDecimal time,
            BigDecimal duration,
            BigDecimal frequency,
            BigDecimal confidence,
            String noteName,
            Integer midiNumber,
            Integer fret,
            Integer stringNumber
    ) {
    }
}

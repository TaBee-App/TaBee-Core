package com.tabee.backend.tab;

import java.math.BigDecimal;
import java.util.List;

public record GeneratedTabResult(
        String sourceAudio,
        String instrument,
        String tuning,
        Integer estimatedTempo,
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

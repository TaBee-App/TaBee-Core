package com.tabee.backend.tab;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.OneToMany;
import jakarta.persistence.Table;

@Entity
@Table(name = "tab_data")
public class TabData {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "tab_data_id")
    private Long id;

    @Column(length = 50)
    private String tuning;

    @Column(name = "estimated_tempo")
    private Integer estimatedTempo;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt = OffsetDateTime.now();

    @OneToMany(mappedBy = "parentTabData", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<NoteEvent> noteEvents = new ArrayList<>();

    public Long getId() {
        return id;
    }

    public String getTuning() {
        return tuning;
    }

    public void setTuning(String tuning) {
        this.tuning = tuning;
    }

    public Integer getEstimatedTempo() {
        return estimatedTempo;
    }

    public void setEstimatedTempo(Integer estimatedTempo) {
        this.estimatedTempo = estimatedTempo;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public List<NoteEvent> getNoteEvents() {
        return noteEvents;
    }

    public void replaceNoteEvents(List<NoteEvent> newNoteEvents) {
        noteEvents.clear();
        if (newNoteEvents == null) {
            return;
        }
        for (NoteEvent noteEvent : newNoteEvents) {
            noteEvent.setParentTabData(this);
            noteEvents.add(noteEvent);
        }
    }
}

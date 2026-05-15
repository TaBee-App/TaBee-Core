package com.tabee.backend.tab;

import java.math.BigDecimal;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

@Entity
@Table(name = "note_events")
public class NoteEvent {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "note_event_id")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "parent_tab_data_id", nullable = false)
    private TabData parentTabData;

    @Column(nullable = false, precision = 10, scale = 4)
    private BigDecimal time;

    @Column(precision = 10, scale = 3)
    private BigDecimal frequency;

    @Column(precision = 5, scale = 4)
    private BigDecimal confidence;

    @Column(name = "note_name", length = 10)
    private String noteName;

    @Column(name = "midi_number")
    private Integer midiNumber;

    private Integer fret;

    @Column(name = "string_number")
    private Integer stringNumber;

    public Long getId() {
        return id;
    }

    public TabData getParentTabData() {
        return parentTabData;
    }

    public void setParentTabData(TabData parentTabData) {
        this.parentTabData = parentTabData;
    }

    public BigDecimal getTime() {
        return time;
    }

    public void setTime(BigDecimal time) {
        this.time = time;
    }

    public BigDecimal getFrequency() {
        return frequency;
    }

    public void setFrequency(BigDecimal frequency) {
        this.frequency = frequency;
    }

    public BigDecimal getConfidence() {
        return confidence;
    }

    public void setConfidence(BigDecimal confidence) {
        this.confidence = confidence;
    }

    public String getNoteName() {
        return noteName;
    }

    public void setNoteName(String noteName) {
        this.noteName = noteName;
    }

    public Integer getMidiNumber() {
        return midiNumber;
    }

    public void setMidiNumber(Integer midiNumber) {
        this.midiNumber = midiNumber;
    }

    public Integer getFret() {
        return fret;
    }

    public void setFret(Integer fret) {
        this.fret = fret;
    }

    public Integer getStringNumber() {
        return stringNumber;
    }

    public void setStringNumber(Integer stringNumber) {
        this.stringNumber = stringNumber;
    }
}

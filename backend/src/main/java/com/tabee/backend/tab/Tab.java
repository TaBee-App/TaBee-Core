package com.tabee.backend.tab;

import java.time.OffsetDateTime;

import com.tabee.backend.audio.AudioFile;
import com.tabee.backend.user.User;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToOne;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;

@Entity
@Table(name = "tabs")
public class Tab {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "tab_id")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "owner_user_id", nullable = false)
    private User owner;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "source_audio_id", nullable = false)
    private AudioFile sourceAudio;

    @OneToOne(fetch = FetchType.LAZY, cascade = CascadeType.ALL, orphanRemoval = true, optional = false)
    @JoinColumn(name = "tab_data_id", nullable = false)
    private TabData tabData;

    @Column(nullable = false)
    private String title;

    private String artist;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt = OffsetDateTime.now();

    @Column(name = "updated_at", nullable = false)
    private OffsetDateTime updatedAt = OffsetDateTime.now();

    @PreUpdate
    void preUpdate() {
        updatedAt = OffsetDateTime.now();
    }

    public Long getId() {
        return id;
    }

    public User getOwner() {
        return owner;
    }

    public void setOwner(User owner) {
        this.owner = owner;
    }

    public AudioFile getSourceAudio() {
        return sourceAudio;
    }

    public void setSourceAudio(AudioFile sourceAudio) {
        this.sourceAudio = sourceAudio;
    }

    public TabData getTabData() {
        return tabData;
    }

    public void setTabData(TabData tabData) {
        this.tabData = tabData;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getArtist() {
        return artist;
    }

    public void setArtist(String artist) {
        this.artist = artist;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public OffsetDateTime getUpdatedAt() {
        return updatedAt;
    }
}

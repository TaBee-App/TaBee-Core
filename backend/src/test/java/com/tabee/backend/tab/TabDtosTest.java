package com.tabee.backend.tab;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.OffsetDateTime;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.tabee.backend.user.User;

import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

class TabDtosTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void mapsTabResponseForOwner() {
        User owner = userWithId(41L, "melis");
        Tab tab = tabWithData(9L, owner, "Song Title", "Artist", "bass", 120);

        TabDtos.TabResponse response = TabDtos.TabResponse.from(tab, 41L, true, 6, "/api/media/avatar.png");

        assertEquals(9L, response.id());
        assertEquals(41L, response.ownerUserId());
        assertEquals("melis", response.ownerUsername());
        assertEquals("/api/media/avatar.png", response.ownerProfileImageUrl());
        assertEquals("Song Title", response.title());
        assertEquals("Artist", response.artist());
        assertEquals("bass", response.tuning());
        assertEquals(120, response.estimatedTempo());
        assertTrue(response.createdByCurrentUser());
        assertTrue(response.favoritedByCurrentUser());
        assertEquals(6, response.favoriteCount());
        assertEquals("E2", response.jsonData().get("notes").get(0).get("note").asText());
    }

    @Test
    void mapsTabResponseForDifferentUser() {
        User owner = userWithId(41L, "melis");
        Tab tab = tabWithData(9L, owner, "Song Title", null, "standard", 90);

        TabDtos.TabResponse response = TabDtos.TabResponse.from(tab, 99L, false, 0, null);

        assertFalse(response.createdByCurrentUser());
        assertFalse(response.favoritedByCurrentUser());
        assertEquals(0, response.favoriteCount());
        assertEquals("standard", response.tuning());
        assertEquals(90, response.estimatedTempo());
    }

    private User userWithId(Long id, String username) {
        User user = new User();
        ReflectionTestUtils.setField(user, "id", id);
        user.setUsername(username);
        user.setEmail(username + "@example.com");
        user.setPasswordHash("hashed-password");
        return user;
    }

    private Tab tabWithData(Long id, User owner, String title, String artist, String tuning, Integer tempo) {
        ObjectNode jsonData = objectMapper.createObjectNode();
        jsonData.putArray("notes")
                .addObject()
                .put("time", 0.0)
                .put("note", "E2");

        TabData tabData = new TabData();
        ReflectionTestUtils.setField(tabData, "id", 33L);
        ReflectionTestUtils.setField(tabData, "createdAt", OffsetDateTime.parse("2026-05-19T12:00:00+03:00"));
        tabData.setTuning(tuning);
        tabData.setEstimatedTempo(tempo);
        tabData.setJsonData(jsonData);

        Tab tab = new Tab();
        ReflectionTestUtils.setField(tab, "id", id);
        ReflectionTestUtils.setField(tab, "createdAt", OffsetDateTime.parse("2026-05-19T12:00:00+03:00"));
        ReflectionTestUtils.setField(tab, "updatedAt", OffsetDateTime.parse("2026-05-19T12:30:00+03:00"));
        tab.setOwner(owner);
        tab.setTabData(tabData);
        tab.setTitle(title);
        tab.setArtist(artist);
        return tab;
    }
}

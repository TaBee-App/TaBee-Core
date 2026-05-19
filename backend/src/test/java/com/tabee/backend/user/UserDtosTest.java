package com.tabee.backend.user;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.OffsetDateTime;

import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

class UserDtosTest {

    @Test
    void mapsPrivateUserProfileToResponse() {
        User user = baseUser(12L);
        user.setEmail("melis@example.com");
        user.setFullName("Melis Onur");
        user.setEmailConfirmed(true);

        UserDtos.UserResponse response = UserDtos.UserResponse.from(user, "/api/media/avatar.png");

        assertEquals(12L, response.id());
        assertEquals("melis", response.username());
        assertEquals("melis@example.com", response.email());
        assertEquals("Melis Onur", response.fullName());
        assertTrue(response.emailConfirmed());
        assertEquals("/api/media/avatar.png", response.profileImageUrl());
    }

    @Test
    void mapsPublicUserProfileWithoutLeakingEmail() {
        User user = baseUser(21L);
        user.setEmail("private@example.com");
        user.setFullName("Public Name");

        UserDtos.PublicUserResponse response = UserDtos.PublicUserResponse.from(
                user,
                "/api/media/public.png",
                false,
                3,
                5
        );

        assertEquals(21L, response.id());
        assertEquals("melis", response.username());
        assertEquals("Public Name", response.fullName());
        assertEquals("/api/media/public.png", response.profileImageUrl());
        assertFalse(response.followedByCurrentUser());
        assertEquals(3, response.followerCount());
        assertEquals(5, response.followingCount());
    }

    private User baseUser(Long id) {
        User user = new User();
        ReflectionTestUtils.setField(user, "id", id);
        ReflectionTestUtils.setField(user, "createdAt", OffsetDateTime.parse("2026-05-19T12:00:00+03:00"));
        ReflectionTestUtils.setField(user, "updatedAt", OffsetDateTime.parse("2026-05-19T12:30:00+03:00"));
        user.setUsername("melis");
        user.setPasswordHash("hashed-password");
        return user;
    }
}

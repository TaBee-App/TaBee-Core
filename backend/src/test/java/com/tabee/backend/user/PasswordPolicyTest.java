package com.tabee.backend.user;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

class PasswordPolicyTest {

    @Test
    void acceptsStrongPassword() {
        assertDoesNotThrow(() ->
                PasswordPolicy.validate("MellowBass!72", "melis", "melis@example.com", "Melis Onur"));
    }

    @Test
    void rejectsPasswordContainingUsername() {
        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                PasswordPolicy.validate("Melis2026!", "melis", "melis@example.com", "Melis Onur"));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Password cannot include your username, email, or name.", exception.getReason());
    }

    @Test
    void rejectsSimpleSequence() {
        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                PasswordPolicy.validate("Abcd1234!", "user", "user@example.com", "User Name"));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Password cannot contain simple keyboard or number sequences.", exception.getReason());
    }

    @Test
    void rejectsPasswordShorterThanEightCharacters() {
        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                PasswordPolicy.validate("A1!tiny", "melis", "melis@example.com", "Melis Onur"));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Password must be at least 8 characters.", exception.getReason());
    }

    @Test
    void rejectsPasswordWithoutSpecialCharacter() {
        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                PasswordPolicy.validate("MellowBass72", "melis", "melis@example.com", "Melis Onur"));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Password must include a special character.", exception.getReason());
    }

    @Test
    void rejectsThreeRepeatedCharacters() {
        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                PasswordPolicy.validate("BeeeeLine!72", "melis", "melis@example.com", "Melis Onur"));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Password cannot repeat the same character 3 times in a row.", exception.getReason());
    }

    @Test
    void rejectsPasswordContainingEmailToken() {
        ResponseStatusException exception = assertThrows(ResponseStatusException.class, () ->
                PasswordPolicy.validate("StudioExample!72", "melis", "studio@example.com", "Melis Onur"));

        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatusCode());
        assertEquals("Password cannot include your username, email, or name.", exception.getReason());
    }
}

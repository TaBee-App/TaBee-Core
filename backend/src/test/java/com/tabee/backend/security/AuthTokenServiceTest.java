package com.tabee.backend.security;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.Optional;

import com.tabee.backend.user.User;
import com.tabee.backend.user.UserRepository;

import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

class AuthTokenServiceTest {

    @Test
    void issuedTokenResolvesBackToUser() {
        UserRepository userRepository = org.mockito.Mockito.mock(UserRepository.class);
        AuthTokenService service = new AuthTokenService(userRepository);
        User user = userWithId(41L);
        when(userRepository.findById(41L)).thenReturn(Optional.of(user));

        String token = service.issueToken(user);

        assertTrue(service.findUserByToken(token).isPresent());
        verify(userRepository).findById(41L);
    }

    @Test
    void unknownTokenReturnsEmptyOptional() {
        UserRepository userRepository = org.mockito.Mockito.mock(UserRepository.class);
        AuthTokenService service = new AuthTokenService(userRepository);

        assertTrue(service.findUserByToken("missing-token").isEmpty());
    }

    @Test
    void issuedTokensAreUnique() {
        UserRepository userRepository = org.mockito.Mockito.mock(UserRepository.class);
        AuthTokenService service = new AuthTokenService(userRepository);
        User user = userWithId(7L);

        String firstToken = service.issueToken(user);
        String secondToken = service.issueToken(user);

        assertNotEquals(firstToken, secondToken);
    }

    private User userWithId(Long id) {
        User user = new User();
        ReflectionTestUtils.setField(user, "id", id);
        return user;
    }
}

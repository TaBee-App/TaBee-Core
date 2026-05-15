package com.tabee.backend.security;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.user.User;

@Component
public class CurrentUser {

    public User require(User user) {
        if (user == null) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Authentication token is missing or invalid");
        }
        return user;
    }
}

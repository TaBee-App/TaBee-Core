package com.tabee.backend.user;

import jakarta.validation.Valid;

import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.RestController;

import com.tabee.backend.security.CurrentUser;
import com.tabee.backend.user.UserDtos.UserResponse;
import com.tabee.backend.user.UserDtos.UserUpdateRequest;

@RestController
@RequestMapping("/api/users")
public class UserController {
    private final UserService userService;
    private final CurrentUser currentUser;

    public UserController(UserService userService, CurrentUser currentUser) {
        this.userService = userService;
        this.currentUser = currentUser;
    }

    @GetMapping("/me")
    public UserResponse me(@AuthenticationPrincipal User user) {
        return UserResponse.from(currentUser.require(user));
    }

    @GetMapping("/{id}")
    public UserResponse findPublicProfile(@PathVariable Long id) {
        return UserResponse.from(userService.findById(id));
    }

    @PutMapping("/me")
    public UserResponse updateMe(@AuthenticationPrincipal User user, @Valid @RequestBody UserUpdateRequest request) {
        return UserResponse.from(userService.update(currentUser.require(user).getId(), request));
    }

    @DeleteMapping("/me")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteMe(@AuthenticationPrincipal User user) {
        userService.delete(currentUser.require(user).getId());
    }
}

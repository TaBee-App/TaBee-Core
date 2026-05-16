package com.tabee.backend.user;

import jakarta.validation.Valid;

import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.RestController;

import com.tabee.backend.security.CurrentUser;
import com.tabee.backend.user.UserDtos.PublicUserResponse;
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

    @GetMapping("/search")
    public java.util.List<PublicUserResponse> search(@AuthenticationPrincipal User user,
                                                     @org.springframework.web.bind.annotation.RequestParam String q) {
        User current = currentUser.require(user);
        return userService.search(q).stream()
                .filter(found -> !found.getId().equals(current.getId()))
                .map(found -> publicUserResponse(current, found))
                .toList();
    }

    @GetMapping("/{id}")
    public PublicUserResponse findPublicProfile(@AuthenticationPrincipal User user, @PathVariable Long id) {
        User current = currentUser.require(user);
        return publicUserResponse(current, userService.findById(id));
    }

    @GetMapping("/{id}/following")
    public java.util.List<PublicUserResponse> publicFollowing(@AuthenticationPrincipal User user, @PathVariable Long id) {
        User current = currentUser.require(user);
        User viewed = userService.findById(id);
        return userService.findFollowing(viewed).stream()
                .map(follow -> publicUserResponse(current, follow.getFollowed()))
                .toList();
    }

    @GetMapping("/{id}/followers")
    public java.util.List<PublicUserResponse> publicFollowers(@AuthenticationPrincipal User user, @PathVariable Long id) {
        User current = currentUser.require(user);
        User viewed = userService.findById(id);
        return userService.findFollowers(viewed).stream()
                .map(follow -> publicUserResponse(current, follow.getFollower()))
                .toList();
    }

    @GetMapping("/me/following")
    public java.util.List<PublicUserResponse> following(@AuthenticationPrincipal User user) {
        User current = currentUser.require(user);
        return userService.findFollowing(current).stream()
                .map(follow -> publicUserResponse(current, follow.getFollowed()))
                .toList();
    }

    @GetMapping("/me/followers")
    public java.util.List<PublicUserResponse> followers(@AuthenticationPrincipal User user) {
        User current = currentUser.require(user);
        return userService.findFollowers(current).stream()
                .map(follow -> PublicUserResponse.from(
                        follow.getFollower(),
                        userService.isFollowing(current, follow.getFollower().getId()),
                        userService.followerCount(follow.getFollower().getId()),
                        userService.followingCount(follow.getFollower().getId())
                ))
                .toList();
    }

    @PostMapping("/{id}/follow")
    public PublicUserResponse follow(@AuthenticationPrincipal User user, @PathVariable Long id) {
        User current = currentUser.require(user);
        return publicUserResponse(current, userService.follow(current, id));
    }

    @DeleteMapping("/{id}/follow")
    public PublicUserResponse unfollow(@AuthenticationPrincipal User user, @PathVariable Long id) {
        User current = currentUser.require(user);
        return publicUserResponse(current, userService.unfollow(current, id));
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

    private PublicUserResponse publicUserResponse(User current, User viewed) {
        return PublicUserResponse.from(
                viewed,
                userService.isFollowing(current, viewed.getId()),
                userService.followerCount(viewed.getId()),
                userService.followingCount(viewed.getId())
        );
    }
}

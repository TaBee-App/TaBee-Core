package com.tabee.backend.security;

import java.io.IOException;
import java.util.List;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import com.tabee.backend.user.User;

@Component
public class TokenAuthenticationFilter extends OncePerRequestFilter {
    private static final String BEARER_PREFIX = "Bearer ";

    private final AuthTokenService authTokenService;

    public TokenAuthenticationFilter(AuthTokenService authTokenService) {
        this.authTokenService = authTokenService;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        String authorization = request.getHeader("Authorization");
        if (authorization != null && authorization.startsWith(BEARER_PREFIX)) {
            String token = authorization.substring(BEARER_PREFIX.length());
            authTokenService.findUserByToken(token).ifPresent(this::authenticate);
        }

        filterChain.doFilter(request, response);
    }

    private void authenticate(User user) {
        UsernamePasswordAuthenticationToken authentication =
                new UsernamePasswordAuthenticationToken(user, null, List.of());
        SecurityContextHolder.getContext().setAuthentication(authentication);
    }
}

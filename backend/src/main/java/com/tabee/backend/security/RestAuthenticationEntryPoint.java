package com.tabee.backend.security;

import java.io.IOException;
import java.time.OffsetDateTime;

import com.fasterxml.jackson.databind.ObjectMapper;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;

@Component
public class RestAuthenticationEntryPoint implements org.springframework.security.web.AuthenticationEntryPoint {
    private final ObjectMapper objectMapper;

    public RestAuthenticationEntryPoint(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @Override
    public void commence(HttpServletRequest request, HttpServletResponse response,
                         org.springframework.security.core.AuthenticationException authException) throws IOException {
        response.setStatus(HttpStatus.UNAUTHORIZED.value());
        response.setContentType("application/json");
        objectMapper.writeValue(response.getWriter(), new SecurityErrorResponse(
                OffsetDateTime.now(),
                HttpStatus.UNAUTHORIZED.value(),
                HttpStatus.UNAUTHORIZED.getReasonPhrase(),
                "Authentication token is missing or invalid",
                request.getRequestURI()
        ));
    }

    public record SecurityErrorResponse(
            OffsetDateTime timestamp,
            int status,
            String error,
            String message,
            String path
    ) {
    }
}

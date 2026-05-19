package com.tabee.backend;

import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import static org.junit.jupiter.api.Assertions.assertTrue;

class TabeeBackendApplicationTests {

    @Test
    void applicationClassIsSpringBootApplication() {
        assertTrue(TabeeBackendApplication.class.isAnnotationPresent(SpringBootApplication.class));
    }
}

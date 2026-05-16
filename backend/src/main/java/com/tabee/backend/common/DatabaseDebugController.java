package com.tabee.backend.common;

import java.util.Map;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/debug/database")
public class DatabaseDebugController {
    private final JdbcTemplate jdbcTemplate;

    public DatabaseDebugController(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @GetMapping
    public Map<String, Object> databaseInfo() {
        return Map.of(
                "database", queryForString("SELECT current_database()"),
                "schema", queryForString("SELECT current_schema()"),
                "user", queryForString("SELECT current_user"),
                "usersCount", count("users"),
                "tabsCount", count("tabs"),
                "tabDataCount", count("tab_data")
        );
    }

    private String queryForString(String sql) {
        return jdbcTemplate.queryForObject(sql, String.class);
    }

    private Long count(String tableName) {
        return jdbcTemplate.queryForObject("SELECT COUNT(*) FROM " + tableName, Long.class);
    }
}

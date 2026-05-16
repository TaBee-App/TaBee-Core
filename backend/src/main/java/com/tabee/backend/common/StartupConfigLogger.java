package com.tabee.backend.common;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

@Component
public class StartupConfigLogger implements ApplicationRunner {
    private static final Logger log = LoggerFactory.getLogger(StartupConfigLogger.class);

    private final Environment environment;

    public StartupConfigLogger(Environment environment) {
        this.environment = environment;
    }

    @Override
    public void run(ApplicationArguments args) {
        String datasourceUrl = environment.getProperty("spring.datasource.url", "");
        String coreRoot = environment.getProperty("tabee.core-root", "");
        String pythonCommand = environment.getProperty("tabee.python-command", "");

        log.info("TaBee config: datasource={}, coreRoot={}, pythonCommand={}",
                sanitizeDatasourceUrl(datasourceUrl),
                coreRoot,
                pythonCommand);
    }

    private String sanitizeDatasourceUrl(String datasourceUrl) {
        if (datasourceUrl == null || datasourceUrl.isBlank()) {
            return "<empty>";
        }
        int queryStart = datasourceUrl.indexOf('?');
        return queryStart >= 0 ? datasourceUrl.substring(0, queryStart) + "?..." : datasourceUrl;
    }
}

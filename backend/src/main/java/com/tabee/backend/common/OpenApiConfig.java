package com.tabee.backend.common;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.security.SecurityRequirement;
import io.swagger.v3.oas.models.security.SecurityScheme;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OpenApiConfig {

    @Bean
    OpenAPI tabeeOpenApi() {
        return new OpenAPI()
                .components(new Components()
                        .addSecuritySchemes("bearerAuth", new SecurityScheme()
                                .type(SecurityScheme.Type.HTTP)
                                .scheme("bearer")
                                .bearerFormat("opaque-token")))
                .addSecurityItem(new SecurityRequirement().addList("bearerAuth"))
                .info(new Info()
                        .title("TaBee Backend API")
                        .version("0.0.1")
                        .description("REST API for users, audio uploads, audio processing bridge, and tabs."));
    }
}

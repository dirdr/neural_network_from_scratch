# Use official Rust image as build stage
FROM rust:1.82 as builder

WORKDIR /app

# Copy the entire project
COPY . .

# Build the application in release mode
RUN cargo build --release

# Use a minimal runtime image
FROM debian:bookworm-slim

# Install required runtime dependencies
RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built binary from builder stage
COPY --from=builder /app/target/release/nn_from_scratch /app/nn_from_scratch

# Copy the MNIST resources
COPY --from=builder /app/mnist/resources /app/mnist/resources

# Create directory for persistent models
RUN mkdir -p /app/models

# Expose the WebSocket port
EXPOSE 8080

# Volume for persistent model storage
VOLUME ["/app/models"]

# Set environment variables for better logging
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# Health check to verify the service is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: run websocket server with CNN enabled
CMD ["./nn_from_scratch", "websocket", "--address", "0.0.0.0:8080", "--with-conv"]
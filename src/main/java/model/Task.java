package model;

public record Task (
    String id,
    long lengthMI,
    int requiredPes,
    long ramMb,
    long diskMb
) {
    public Task {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("Task ID is required");
        }

        if (lengthMI <= 0) {
            throw new IllegalArgumentException("Task length must be positive");
        }

        if (requiredPes <= 0) {
            throw new IllegalArgumentException("Required PE's must be positive");
        }

        if (ramMb <= 0 || diskMb <= 0) {
            throw new IllegalArgumentException("Resource demands cannot be negative");
        }
    }
}
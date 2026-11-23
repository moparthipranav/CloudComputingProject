package model;

public record DataDependency(
    String sourceTaskId,
    String targetTaskId,
    long dataBytes
) {
    public DataDependency {
        if (sourceTaskId == null || sourceTaskId.isBlank()
                || targetTaskId == null || targetTaskId.isBlank()) {
            throw new IllegalArgumentException("Task IDs are required");
        }
        if (sourceTaskId.equals(targetTaskId)) {
            throw new IllegalArgumentException(
                    "A task cannot depend on itself");
        }
        if (dataBytes < 0) {
            throw new IllegalArgumentException(
                    "Data volume cannot be negative");
        }
    }
}
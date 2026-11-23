package model;

import java.util.*;

public final class Workflow {

    private final Map<String, Task> tasks = new LinkedHashMap<>(); // maintains the insertion order of the tasks
    private final List<DataDependency> dependencies = new ArrayList<>();

    public Workflow addTask(Task task) {
        Objects.requireNonNull(task, "task cannot be null");

        if (tasks.putIfAbsent(task.id(), task) != null) {
            throw new IllegalArgumentException(
                "Duplicate Task Id: " + task.id()
            );
        }

        return this; // returns the whole workflow object with the new task added
    }

    public Workflow addDependency(DataDependency dependency) {
        Objects.requireNonNull(
            dependency, "Dependency cannot be null"
        );

        dependencies.add(dependency);
        return this;
    }

    public Map<String, Task> tasks() {
        return Collections.unmodifiableMap(tasks);
    }

    public List<DataDependency> dependencies() {
        return Collections.unmodifiableList(dependencies);
    }

    public List<Task> entryTasks() { // adds a set of entry tasks (which do not have a parent)
        Set<String> targets = new HashSet<>();

        for (DataDependency dependency : dependencies) {
            targets.add(dependency.targetTaskId());
        }

        return tasks.values().stream()
                .filter(task -> !targets.contains(task.id()))
                .toList();
    }

    public List<Task> exitTasks() { // adds a set of exit tasks which do not have a child
        Set<String> sources = new HashSet<>();

        for (DataDependency dependency: dependencies) {
            sources.add(dependency.sourceTaskId());
        }

        return tasks.values().stream()
                .filter(task -> !sources.contains(task.id()))
                .toList();
    }

    public List<Task> topologicalOrder() {
        // using Kahn's algorithm for topological ordering of the nodes.
        validateDependencies();

        Map<String, Integer> indegree = new LinkedHashMap<>();
        Map<String, List<String>> children = new HashMap<>();

        for (String id: tasks.keySet()) {
            indegree.put(id, 0);
            children.put(id, new ArrayList<>());
        }

        for (DataDependency dependency : dependencies) {
            children.get(dependency.sourceTaskId()).add(dependency.targetTaskId());

            indegree.merge(dependency.targetTaskId(), 1, Integer::sum);
        }

        Queue<String> ready = new ArrayDeque<>();

        for (String id : tasks.keySet()) {
            if (indegree.get(id) == 0) {
                ready.add(id);
            }
        }

        List<Task> order = new ArrayList<>();

        while(!ready.isEmpty()) {
            String id = ready.remove();
            order.add(tasks.get(id));

            for (String child: children.get(id)) {
                int remaining = indegree.merge(child, -1, Integer::sum);

                if (remaining == 0) {
                    ready.add(child);
                }
            }
        }

        if (order.size() != tasks.size()) {
            throw new IllegalStateException(
                "Workflow contains a cycle"
            );
        }

        return List.copyOf(order);
    }

    private void validateDependencies() {
        for (DataDependency dependency : dependencies) {
            if (!tasks.containsKey(dependency.sourceTaskId())) {
                throw new IllegalStateException(
                        "Unknown source task: "
                                + dependency.sourceTaskId());
            }

            if (!tasks.containsKey(dependency.targetTaskId())) {
                throw new IllegalStateException(
                        "Unknown target task: "
                                + dependency.targetTaskId());
            }
        }
    }

    public long totalComputeMI() {
        long total = 0;
        for (Task task : tasks.values()) {
            total = Math.addExact(total, task.lengthMI());
        }
        return total;
    }

    public long totalDataBytes() {
        long total = 0;
        for (DataDependency dependency : dependencies) {
            total = Math.addExact(total, dependency.dataBytes());
        }
        return total;
    }
}
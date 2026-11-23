import infrastructure.NetworkTopologyModel;
import model.DataDependency;
import model.Task;
import model.Workflow;
import scheduling.WorkflowScheduler;
import scheduling.WorkflowScheduler.ScheduleResult;
import scheduling.WorkflowScheduler.TaskSchedule;
import scheduling.WorkflowScheduler.VmResource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        Path topologyFile = Path.of("src/main/java/infrastructure/topology.brite")
            .toAbsolutePath().normalize();
        if (!Files.isRegularFile(topologyFile)) {
            throw new IllegalStateException("BRITE topology file not found: " + topologyFile);
        }

        Workflow workflow = createWorkflow();
        List<VmResource> vms = List.of(
            new VmResource("VM0", 1000, 4, 4096, 20_000, "21"),
            new VmResource("VM1", 1000, 4, 4096, 20_000, "25"),
            new VmResource("VM2", 1000, 4, 4096, 20_000, "29"),
            new VmResource("VM3", 1000, 4, 4096, 20_000, "33")
        );

        NetworkTopologyModel topology = new NetworkTopologyModel(topologyFile.toString());
        ScheduleResult result = new WorkflowScheduler(topology, vms).schedule(workflow);
        validateSchedule(workflow, vms, result, topology);

        System.out.println("Workflow execution order: "
            + workflow.topologicalOrder().stream().map(Task::id).toList());
        System.out.printf(Locale.ROOT, "Makespan: %.6f seconds%n", result.makespan());
        System.out.printf(Locale.ROOT, "Communication wait: %.6f seconds%n%n",
            result.totalCommunicationWait());

        System.out.println("Task schedule:");
        System.out.printf("%-8s %-8s %-12s %-12s %-12s %-12s%n",
            "Task", "VM", "Start(s)", "Finish(s)", "Exec(s)", "Comm wait(s)");
        for (TaskSchedule task : result.taskSchedules().values()) {
            System.out.printf(Locale.ROOT, "%-8s %-8s %-12.6f %-12.6f %-12.6f %-12.6f%n",
                task.taskId(), task.vmId(), task.startTime(), task.finishTime(),
                task.executionTime(), task.communicationWait());
        }

        printCommunicationCostMatrix(workflow, vms, result, topology);
    }

    private static Workflow createWorkflow() {
        return new Workflow()
            .addTask(new Task("T1", 10_000, 1, 512, 100))
            .addTask(new Task("T2", 20_000, 1, 512, 100))
            .addTask(new Task("T3", 15_000, 1, 1024, 200))
            .addTask(new Task("T4", 8_000, 1, 512, 100))
            .addDependency(new DataDependency("T1", "T2", 50_000_000))
            .addDependency(new DataDependency("T1", "T3", 10_000_000))
            .addDependency(new DataDependency("T2", "T4", 20_000_000))
            .addDependency(new DataDependency("T3", "T4", 5_000_000));
    }

    private static void validateSchedule(
        Workflow workflow,
        List<VmResource> vms,
        ScheduleResult result,
        NetworkTopologyModel topology
    ) {
        if (result.taskSchedules().size() != workflow.tasks().size()) {
            throw new IllegalStateException("Not every workflow task was scheduled");
        }

        Map<String, VmResource> vmById = new LinkedHashMap<>();
        vms.forEach(vm -> vmById.put(vm.id(), vm));

        Map<String, Double> vmAvailableAt = new HashMap<>();
        for (TaskSchedule schedule : result.taskSchedules().values()) {
            Task task = workflow.tasks().get(schedule.taskId());
            if (task == null) {
                throw new IllegalStateException("Unknown task in schedule: " + schedule.taskId());
            }
            VmResource vm = vmById.get(schedule.vmId());
            if (vm == null || !vm.canRun(task)) {
                throw new IllegalStateException("Task assigned to an incompatible VM: " + task.id());
            }
            if (!Double.isFinite(schedule.startTime()) || !Double.isFinite(schedule.finishTime())
                    || schedule.startTime() < 0 || schedule.finishTime() <= schedule.startTime()
                    || schedule.startTime() + 1e-9 < vmAvailableAt.getOrDefault(schedule.vmId(), 0.0)) {
                throw new IllegalStateException("Invalid or overlapping task schedule: " + task.id());
            }
            vmAvailableAt.put(schedule.vmId(), schedule.finishTime());
        }

        for (DataDependency edge : workflow.dependencies()) {
            TaskSchedule source = result.taskSchedules().get(edge.sourceTaskId());
            TaskSchedule target = result.taskSchedules().get(edge.targetTaskId());
            double transferTime = source.vmId().equals(target.vmId())
                ? 0
                : topology.getPathMetrics(source.hostId(), target.hostId(), edge.dataBytes())
                    .transferTimeSeconds();
            if (target.startTime() + 1e-9 < source.finishTime() + transferTime) {
                throw new IllegalStateException(
                    "Dependency is violated by the schedule: "
                        + edge.sourceTaskId() + " -> " + edge.targetTaskId()
                );
            }
        }
    }

    private static void printCommunicationCostMatrix(
        Workflow workflow,
        List<VmResource> vms,
        ScheduleResult result,
        NetworkTopologyModel topology
    ) {
        Map<String, Integer> vmIndexes = new LinkedHashMap<>();
        for (int i = 0; i < vms.size(); i++) {
            vmIndexes.put(vms.get(i).id(), i);
        }

        double[][] costsSeconds = new double[vms.size()][vms.size()];
        for (DataDependency edge : workflow.dependencies()) {
            TaskSchedule source = result.taskSchedules().get(edge.sourceTaskId());
            TaskSchedule target = result.taskSchedules().get(edge.targetTaskId());
            if (!source.vmId().equals(target.vmId())) {
                costsSeconds[vmIndexes.get(source.vmId())][vmIndexes.get(target.vmId())]
                    += topology.getPathMetrics(source.hostId(), target.hostId(), edge.dataBytes())
                        .transferTimeSeconds();
            }
        }

        System.out.println();
        System.out.println("Communication cost matrix (aggregate transfer time, seconds)");
        System.out.println("Rows = source VM; columns = destination VM.");
        System.out.printf("%12s", "");
        for (VmResource vm : vms) {
            System.out.printf("%12s", vm.id());
        }
        System.out.println();

        for (int row = 0; row < vms.size(); row++) {
            System.out.printf("%12s", vms.get(row).id());
            for (int column = 0; column < vms.size(); column++) {
                if (!Double.isFinite(costsSeconds[row][column]) || costsSeconds[row][column] < 0) {
                    throw new IllegalStateException("Invalid communication cost in VM matrix");
                }
                System.out.printf(Locale.ROOT, "%12.6f", costsSeconds[row][column]);
            }
            System.out.println();
        }
    }
}
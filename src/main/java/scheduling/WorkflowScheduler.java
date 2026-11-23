package scheduling;

import model.Task;
import infrastructure.NetworkTopologyModel;
import model.Workflow;
import model.DataDependency;

import java.util.*;

/** Greedily places topologically ordered tasks on the VM that yields the earliest finish. */
public final class WorkflowScheduler {

    public record VmResource (
        String id,
        double mipsPerCore,
        int pes,
        long ramMb,
        long storageMb,
        String hostId
    ) {
        public VmResource {
            if (id == null || id.isBlank() || hostId == null || hostId.isBlank()) {
                throw new IllegalArgumentException("VM ID and host ID are required");
            }

            if (!Double.isFinite(mipsPerCore) || mipsPerCore <= 0 || pes <= 0 || ramMb <= 0 || storageMb <= 0) {
                throw new IllegalArgumentException("VM resource capacities must be positive");
            }
        }

        public boolean canRun(Task task) {
            Objects.requireNonNull(task, "task is required");
            return pes >= task.requiredPes()
                && ramMb >= task.ramMb()
                && storageMb >= task.diskMb();
        }
    }

    public record TaskSchedule(
        String taskId,
        String vmId,
        String hostId,
        double startTime,
        double finishTime,
        double executionTime,
        double communicationWait
    ) {}

    public record ScheduleResult(
        Map<String, TaskSchedule> taskSchedules,
        double makespan,
        double totalCommunicationWait
    ) {
        public ScheduleResult {
            taskSchedules = Collections.unmodifiableMap(
                new LinkedHashMap<>(taskSchedules)
            );
        }
    }

    private final NetworkTopologyModel topology;
    private final List<VmResource> vms;

    public WorkflowScheduler(
        NetworkTopologyModel topology,
        List<VmResource> vms
    ) {
        this.topology = Objects.requireNonNull(topology);
        this.vms = List.copyOf(vms);

        if (this.vms.isEmpty()) {
            throw new IllegalArgumentException(
                "Atleast one VM is required"
            );
        }

        Set<String> vmIds = new HashSet<>();

        for (VmResource vm : this.vms) {
            if (!vmIds.add(vm.id())) {
                throw new IllegalArgumentException("Duplicate VM ID: " + vm.id());
            }
        }
    }


    public ScheduleResult schedule(Workflow workflow) {
        Objects.requireNonNull(workflow);

        List<Task> orderedTasks = workflow.topologicalOrder();

        Map<String, List<DataDependency>> incoming = new HashMap<>();

        for (Task task: orderedTasks) {
            incoming.put(task.id(), new ArrayList<>());
        }

        for (DataDependency edge: workflow.dependencies()) {
            incoming.get(edge.targetTaskId()).add(edge);
        }

        Map<String, TaskSchedule> schedules = new LinkedHashMap<>();
        Map<String, Double> vmAvailableAt = new HashMap<>();

        for (VmResource vm: vms) {
            vmAvailableAt.put(vm.id(), 0.0);
        }

        double totalCommunicationWait = 0.0;

        for (Task task: orderedTasks) {
            VmResource bestVm = null;
            double bestStart = 0.0;
            double bestFinish = Double.POSITIVE_INFINITY;
            double bestCommunicationWait = 0.0;

            for (VmResource candidate: vms) {
                if (!candidate.canRun(task)) {
                    continue;
                }

                double parentReadyAt = 0.0;
                double dataReadyAt = 0.0;

                for (DataDependency edge : incoming.get(task.id())) {
                    TaskSchedule parent = schedules.get(edge.sourceTaskId());

                    if (parent == null) {
                        throw new IllegalStateException("Parent task not scheduled: " + edge.sourceTaskId());
                    }

                    double transferTime = 0.0;

                    if (!parent.vmId().equals(candidate.id())) {
                        NetworkTopologyModel.PathMetrics path =
                            topology.getPathMetrics(
                                parent.hostId(),
                                candidate.hostId(),
                                edge.dataBytes()
                            );
                        transferTime = path.transferTimeSeconds();
                    }

                    parentReadyAt = Math.max(parentReadyAt, parent.finishTime());
                    dataReadyAt = Math.max(dataReadyAt, parent.finishTime() + transferTime);
                }

                double startTime = Math.max(vmAvailableAt.get(candidate.id()), dataReadyAt);

                double effectiveMips =
                        candidate.mipsPerCore() * task.requiredPes();

                double executionTime =
                        task.lengthMI() / effectiveMips;
                if (!Double.isFinite(executionTime) || executionTime <= 0) {
                    throw new IllegalStateException(
                        "Invalid execution time for task " + task.id() + " on VM " + candidate.id()
                    );
                }

                double finishTime = startTime + executionTime;
                if (!Double.isFinite(finishTime)) {
                    throw new IllegalStateException(
                        "Invalid finish time for task " + task.id() + " on VM " + candidate.id()
                    );
                }

                if (finishTime < bestFinish) {
                    bestVm = candidate;
                    bestStart = startTime;
                    bestFinish = finishTime;
                    bestCommunicationWait = Math.max(0.0, dataReadyAt - parentReadyAt);
                }
            }


            if (bestVm == null) {
                throw new IllegalStateException("No VM can satisfy task requirements: " + task.id());
            }

            double executionTime = bestFinish - bestStart;

            TaskSchedule result = new TaskSchedule(
                    task.id(),
                    bestVm.id(),
                    bestVm.hostId(),
                    bestStart,
                    bestFinish,
                    executionTime,
                    bestCommunicationWait
            );

            schedules.put(task.id(), result);

            vmAvailableAt.put(bestVm.id(), bestFinish);
            totalCommunicationWait += bestCommunicationWait;
        }


        double makespan = schedules.values().stream()
                .mapToDouble(TaskSchedule::finishTime)
                .max()
                .orElse(0.0);

        return new ScheduleResult(
                schedules,
                makespan,
                totalCommunicationWait
        );
    }
}
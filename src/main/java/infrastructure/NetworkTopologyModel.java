package infrastructure;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.SimEntity;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.network.topologies.BriteNetworkTopology;
import org.cloudsimplus.network.topologies.NetworkTopology;
import org.cloudsimplus.network.topologies.TopologicalLink;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;

public class NetworkTopologyModel {

    // Node ids in topology.brite (0 = gateway, 1 = first core switch)
    public static final int BROKER_NODE = 0;
    public static final int DATACENTER_NODE = 1;

    private final BriteNetworkTopology topology;
    private final Map<Integer, List<Link>> linksByNode = new HashMap<>();
    private final Set<Integer> nodeIds = new HashSet<>();
    private final Map<NodePair, Route> routeCache = new HashMap<>();

    public record PathMetrics(
        double latencySeconds,
        double bandwidthMbps,
        double transferTimeSeconds
    ) {
        public PathMetrics {
            if (!Double.isFinite(latencySeconds) || latencySeconds < 0
                    || !Double.isFinite(bandwidthMbps) || bandwidthMbps < 0
                    || !Double.isFinite(transferTimeSeconds) || transferTimeSeconds < 0) {
                throw new IllegalArgumentException("Path metrics must be finite and non-negative");
            }
        }
    }

    private record Link(int nodeId, double delaySeconds, double bandwidthMbps) {}
    private record NodePair(int first, int second) {}
    private record Route(double latencySeconds, double bandwidthMbps) {}
    private record QueueEntry(int nodeId, double latencySeconds, double bandwidthMbps) {}

    public NetworkTopologyModel(String briteFile) {
        this.topology = BriteNetworkTopology.getInstance(
            Objects.requireNonNull(briteFile, "BRITE topology file is required")
        );

        topology.getGraph().getNodeList().forEach(node -> nodeIds.add(node.getId()));

        for (TopologicalLink link : topology.getGraph().getLinksList()) {
            if (!Double.isFinite(link.getLinkDelay()) || link.getLinkDelay() < 0
                    || !Double.isFinite(link.getLinkBw()) || link.getLinkBw() <= 0) {
                throw new IllegalArgumentException("Topology links must have non-negative delay and positive bandwidth");
            }

            linksByNode.computeIfAbsent(link.getSrcNodeID(), ignored -> new ArrayList<>())
                .add(new Link(link.getDestNodeID(), link.getLinkDelay(), link.getLinkBw()));
            linksByNode.computeIfAbsent(link.getDestNodeID(), ignored -> new ArrayList<>())
                .add(new Link(link.getSrcNodeID(), link.getLinkDelay(), link.getLinkBw()));
        }
    }

    /** Call after the Datacenter and Broker are created, before simulation.start() */
    public void apply(CloudSimPlus simulation, Datacenter datacenter, DatacenterBroker broker) {
        Objects.requireNonNull(simulation, "simulation is required").setNetworkTopology(topology);
        topology.mapNode((SimEntity)datacenter, DATACENTER_NODE);
        topology.mapNode((SimEntity)broker, BROKER_NODE);
    }

    /**
     * Computes communication time over the minimum-latency BRITE path.
     * BRITE delays are seconds and bandwidths are Mbps.
     */
    public PathMetrics getPathMetrics(String sourceHostId, String destinationHostId, long dataBytes) {
        if (dataBytes < 0) {
            throw new IllegalArgumentException("Data volume cannot be negative");
        }

        int source = parseNodeId(sourceHostId);
        int destination = parseNodeId(destinationHostId);
        if (!nodeIds.contains(source) || !nodeIds.contains(destination)) {
            throw new IllegalArgumentException(
                "Unknown BRITE node in host path: " + sourceHostId + " -> " + destinationHostId
            );
        }

        if (source == destination) {
            return new PathMetrics(0, 0, 0);
        }

        NodePair pair = new NodePair(Math.min(source, destination), Math.max(source, destination));
        Route route = routeCache.computeIfAbsent(pair, ignored -> findRoute(source, destination));
        double transmissionSeconds = ((double) dataBytes * Byte.SIZE)
            / (route.bandwidthMbps() * 1_000_000);
        double transferTimeSeconds = route.latencySeconds() + transmissionSeconds;

        return new PathMetrics(
            route.latencySeconds(),
            route.bandwidthMbps(),
            transferTimeSeconds
        );
    }

    private int parseNodeId(String hostId) {
        Objects.requireNonNull(hostId, "Host ID is required");
        try {
            return Integer.parseInt(hostId);
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(
                "Host ID must be a numeric BRITE node ID: " + hostId, exception
            );
        }
    }

    private Route findRoute(int source, int destination) {
        Map<Integer, Route> bestRoutes = new HashMap<>();
        PriorityQueue<QueueEntry> queue = new PriorityQueue<>(
            Comparator.comparingDouble(QueueEntry::latencySeconds)
                .thenComparing(Comparator.comparingDouble(QueueEntry::bandwidthMbps).reversed())
        );
        bestRoutes.put(source, new Route(0, Double.POSITIVE_INFINITY));
        queue.add(new QueueEntry(source, 0, Double.POSITIVE_INFINITY));

        while (!queue.isEmpty()) {
            QueueEntry current = queue.remove();
            Route knownRoute = bestRoutes.get(current.nodeId());
            if (Double.compare(current.latencySeconds(), knownRoute.latencySeconds()) != 0
                    || Double.compare(current.bandwidthMbps(), knownRoute.bandwidthMbps()) != 0) {
                continue;
            }

            if (current.nodeId() == destination) {
                return knownRoute;
            }

            for (Link link : linksByNode.getOrDefault(current.nodeId(), List.of())) {
                double candidateLatency = current.latencySeconds() + link.delaySeconds();
                double candidateBandwidth = Math.min(current.bandwidthMbps(), link.bandwidthMbps());
                Route previous = bestRoutes.get(link.nodeId());

                if (previous == null || candidateLatency < previous.latencySeconds()
                        || (Double.compare(candidateLatency, previous.latencySeconds()) == 0
                            && candidateBandwidth > previous.bandwidthMbps())) {
                    Route candidate = new Route(candidateLatency, candidateBandwidth);
                    bestRoutes.put(link.nodeId(), candidate);
                    queue.add(new QueueEntry(link.nodeId(), candidateLatency, candidateBandwidth));
                }
            }
        }

        throw new IllegalStateException(
            "No network route between BRITE nodes " + source + " and " + destination
        );
    }

    public NetworkTopology getTopology() { // Check this once
        return topology;
    }
}

package infrastructure;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;

import java.util.ArrayList;
import java.util.List;

public final class HostFactory {

    private HostFactory() {}

    public static Host createHost(
        int hostId,
        int cores,
        double mipsPerCore,
        long ramMb,
        long bandwidthMbps,
        long storageMb
    ) {
        if (hostId < 0 || cores <= 0 || mipsPerCore <= 0 || ramMb <= 0 || bandwidthMbps <= 0 || storageMb <= 0) {
            throw new IllegalArgumentException("Host ID must be non-negative and resource capacities must be positive");
        }

        List<Pe> processingElements = new ArrayList<>();

        for (int i = 0; i < cores; i++) {
            processingElements.add(new PeSimple(mipsPerCore));
        }

        Host host = new HostSimple(
            ramMb,
            bandwidthMbps,
            storageMb,
            processingElements
        );

        host.setId(hostId);
        return host;
    }

    public static List<Host> createHosts(
        int numberOfHosts,
        int coresPerHost,
        double mipsPerCore,
        long ramMb,
        long bandwidthMbps,
        long storageMb
    ) {
        if (numberOfHosts <= 0) {
            throw new IllegalArgumentException("Number of hosts must be positive");
        }

        List<Host> hosts = new ArrayList<>();

        for (int i = 0; i < numberOfHosts; i++) {
            hosts.add(createHost(
                i, coresPerHost, mipsPerCore, ramMb, bandwidthMbps, storageMb
            ));
        }

        return hosts;
    }
}
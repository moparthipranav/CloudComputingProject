/*
 * CloudSimValidator - applies Python-generated VM->host mappings in CloudSim Plus.
 *
 * STATUS: UNTESTED TEMPLATE. It was written without a CloudSim Plus jar or network access
 * available, so it has NOT been compiled or run. It targets the CloudSim Plus 7.x/8.x core
 * API as recalled; package names (org.cloudsimplus.* vs org.cloudbus.cloudsim.*) and a few
 * method signatures differ between releases - fix imports/signatures against the version in
 * your pom.xml. See README section "CloudSim integration".
 *
 * What it does:  reads hosts.csv / vms.csv / mapping_*.csv exported by main.py, builds one
 * datacenter with those hosts, forces each VM onto its mapped host, runs one CPU-bound
 * cloudlet per VM, and writes per-mapping summary metrics to cloudsim_summary.csv.
 *
 * What it does NOT do: model switch-level network latency. Plain CloudSim Plus hosts have no
 * topology-aware inter-VM latency. To measure it you must extend this with the
 * NetworkHost/NetworkVm/NetworkCloudlet classes (Send/Receive tasks) using traffic.csv and
 * topology_edges.csv. Until then, latency/communication numbers remain Python-model values.
 *
 * Usage: java -cp <cloudsimplus-jar-and-deps>:. CloudSimValidator <export_dir>
 */
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

public class CloudSimValidator {
    private static final long CLOUDLET_LENGTH_MI = 100_000;

    /** Allocation policy that places each VM on the host given by the Python mapping. */
    static class MappedAllocationPolicy extends VmAllocationPolicyAbstract {
        private final Map<Long, Integer> vmToHost;
        MappedAllocationPolicy(Map<Long, Integer> vmToHost) { this.vmToHost = vmToHost; }

        @Override
        protected Optional<Host> defaultFindHostForVm(final Vm vm) {
            Integer h = vmToHost.get(vm.getId());
            if (h == null) return Optional.empty();
            Host host = getHostList().get(h);
            return host.isSuitableForVm(vm) ? Optional.of(host) : Optional.empty();
        }
    }

    static List<String[]> readCsv(Path p) throws IOException {
        List<String> lines = Files.readAllLines(p);
        return lines.stream().skip(1).map(l -> l.split(",")).collect(Collectors.toList());
    }

    public static void main(String[] args) throws IOException {
        Path dir = Paths.get(args.length > 0 ? args[0] : "data/results/cloudsim");
        List<String[]> hostRows = readCsv(dir.resolve("hosts.csv"));
        List<String[]> vmRows = readCsv(dir.resolve("vms.csv"));
        List<Path> mappings;
        try (var s = Files.list(dir)) {
            mappings = s.filter(p -> p.getFileName().toString().startsWith("mapping_")).sorted()
                        .collect(Collectors.toList());
        }
        List<String> out = new ArrayList<>();
        out.add("mapping,vms_placed,vms_failed,hosts_used,makespan_s,mean_cpu_util_used_hosts");

        for (Path mapping : mappings) {
            Map<Long, Integer> vmToHost = new HashMap<>();
            for (String[] r : readCsv(mapping)) vmToHost.put(Long.parseLong(r[0]), Integer.parseInt(r[1]));

            CloudSimPlus sim = new CloudSimPlus();
            List<Host> hosts = new ArrayList<>();
            for (String[] r : hostRows) {  // host_id,name,pes,mips_per_pe,ram_mb,bw_mbps,storage_mb
                List<Pe> pes = new ArrayList<>();
                for (int i = 0; i < Integer.parseInt(r[2]); i++) pes.add(new PeSimple(Double.parseDouble(r[3])));
                hosts.add(new HostSimple(Long.parseLong(r[4]), Long.parseLong(r[5]),
                                         Long.parseLong(r[6]), pes));
            }
            new DatacenterSimple(sim, hosts, new MappedAllocationPolicy(vmToHost));
            DatacenterBroker broker = new DatacenterBrokerSimple(sim);

            List<Vm> vms = new ArrayList<>();
            List<Cloudlet> cloudlets = new ArrayList<>();
            for (String[] r : vmRows) {    // vm_id,pes,mips_per_pe,ram_mb,bw_mbps,storage_mb
                Vm vm = new VmSimple(Double.parseDouble(r[2]), Integer.parseInt(r[1]));
                vm.setRam(Long.parseLong(r[3])).setBw(Long.parseLong(r[4])).setSize(Long.parseLong(r[5]));
                vms.add(vm);
                Cloudlet c = new CloudletSimple(CLOUDLET_LENGTH_MI, Integer.parseInt(r[1]),
                                                new UtilizationModelFull());
                c.setVm(vm);
                cloudlets.add(c);
            }
            broker.submitVmList(vms);
            broker.submitCloudletList(cloudlets);
            sim.start();

            long failed = broker.getVmFailedList().size();
            Set<Long> used = new HashSet<>();
            for (Vm vm : broker.getVmCreatedList()) used.add(vm.getHost().getId());
            double makespan = cloudlets.stream().mapToDouble(Cloudlet::getFinishTime).max().orElse(0);
            double util = used.stream().mapToDouble(id -> hosts.get((int) (long) id)
                    .getCpuPercentUtilization()).average().orElse(0);
            out.add(String.format(Locale.ROOT, "%s,%d,%d,%d,%.3f,%.4f",
                    mapping.getFileName(), broker.getVmCreatedList().size(), failed,
                    used.size(), makespan, util));
        }
        Files.write(dir.resolve("cloudsim_summary.csv"), out);
        out.forEach(System.out::println);
    }
}

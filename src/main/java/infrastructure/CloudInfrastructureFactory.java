package infrastructure;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;

import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerTimeShared;

import java.util.ArrayList;
import java.util.List;

/**Creates the actual CloudSimPlus objects for a basic one-datacenter scenario */
public final class CloudInfrastructureFactory {

    private CloudInfrastructureFactory() {}

    
    public static List<Vm> createVms(
        int vmCount, int coresPerVm, double mipsPerCore,
        long ramMbPerVm, long bwMbpsPerVm, long storageMbPerVm
    ) {
        if (vmCount <= 0 || coresPerVm <= 0 || mipsPerCore <= 0) {
            throw new IllegalArgumentException("VM count, core count and MIPS must be positive");
        }

        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < vmCount; i++) {
            Vm vm = new VmSimple(mipsPerCore, coresPerVm, new CloudletSchedulerTimeShared())
                    .setRam(ramMbPerVm)
                    .setBw(bwMbpsPerVm)
                    .setSize(storageMbPerVm);
            vm.setId(i);
            vms.add(vm);
        }

        return vms;
    }


    public static Datacenter createDatacenter(CloudSimPlus simulation, List<Host> hosts) {
        return new DatacenterSimple(simulation, hosts);
    }

    public static DatacenterBroker createBroker(CloudSimPlus simulation) {
        return new DatacenterBrokerSimple(simulation);
    }
}
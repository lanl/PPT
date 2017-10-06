import numpy.lib.recfunctions as rf
from numpy.lib.recfunctions import merge_arrays

dat_fn = '../app_runs/polybench_byfl_20151028.csv'
print 'Reading data from ', dat_fn
junk = genfromtxt(dat_fn, names=True, delimiter=',', skip_footer=6100)
junk = junk.dtype
typ =  dtype([('nameA', 'S40'), ('integer_ops', '<f8'), ('flops', '<f8'), ('FAdd', '<f8'), ('FMul', '<f8'), ('memory_ops', '<f8'), ('loads', '<f8'), ('stores', '<f8'), ('branch_ops', '<f8'), ('uncond_branch_ops', '<f8'), ('cond_branch_ops', '<f8'), ('comparison', '<f8'), ('cpu_ops', '<f8'), ('total_ops', '<f8'), ('vector_ops', '<f8')])
byfl = genfromtxt(dat_fn, delimiter=',', dtype=typ, skip_header=1)

dat_fn = '../app_runs/polybench_mustang_20151110.csv'
print 'Reading data from ', dat_fn
junk = genfromtxt(dat_fn, names=True, delimiter=',', skip_footer=6100)
junk = junk.dtype
typ = dtype([('name', 'S40'), ('EDS', '<f8'), ('edsL1_cache', '<f8'), ('edsL2_cache', '<f8'), ('edsL3_cache', '<f8'), ('PAPI_L1_DCM', '<f8'), ('PAPI_L2_DCM', '<f8'), ('PAPI_L1_TCM', '<f8'), ('PAPI_L2_TCM', '<f8'), ('PAPI_TLB_DM', '<f8'), ('PAPI_BR_TKN', '<f8'), ('PAPI_BR_MSP', '<f8'), ('PAPI_L2_DCA', '<f8'), ('PAPI_L2_TCA', '<f8'), ('Runtime_secs', '<f8')])
data = genfromtxt(dat_fn, delimiter=',', dtype=typ, skip_header=1)

allvars = array(data.dtype.names)
urn, urn_indx = unique(data[allvars[0]], return_inverse=True)
for i in range(len(urn)):
    same_rn = data[urn_indx==i]
    if len(same_rn)>1:
        print urn[i], len(same_rn)

runbase, runN = zip(*[name.split('N') for name in data[allvars[0]]])
runbase = [rn.split('_')[0] for rn in runbase]
urb = unique(runbase)

Ns = [array(x.split('_')).astype(int) for x in runN]
ndim = array([len(x) for x in Ns]).max()
Ns = array([list(x) + [1]*(ndim-len(x)) for x in Ns]).T

Nprod = [prod(array(x.split('_')).astype(int)) for x in runN]

byfl = rf.append_fields(byfl, 'runbase', data=runbase)
byfl = rf.append_fields(byfl, 'Nprod', data=Nprod)

for i in range(3):
    byfl = rf.append_fields(byfl, 'N%d'%i, data=Ns[i])

alldata = merge_arrays((byfl, data), asrecarray=True, flatten=True)
allvars = array(alldata.dtype.names)

for run in urb[:]:
    figure(run); clf()
    indx = alldata['runbase']==run
    n0 = log(alldata['N0'][indx])
    n1 = log(alldata['N1'][indx])
    n2 = log(alldata['N2'][indx])
    cn = ones(len(n0))
    X = column_stack((cn, n0, n1, n2))
    for varnum in range(1, 5) + range(6, 8) + range(9, 13) + range(21, 22):
        y = log(alldata[allvars[varnum]][indx])
        if y.min() != -inf:
            coeffs = linalg.lstsq(X, y)[0]
            print coeffs
            clbl = ': ' + r'$%.2g N_x^%d N_y^%d N_z^%d$'\
              %((exp(coeffs[0]),) + tuple([int(rint(x)) for x in coeffs[1:]]))

            '''
            n0loc = array([n0.min(), n0.max()])
            n1val = n1[1]
            n2val = n2[1]
            yfit = coeffs[0] + coeffs[1]*n0loc + coeffs[2]*n1val + coeffs[3]*n2val
            loglog(exp(n0loc), exp(yfit))
            loglog(alldata['N0'][indx], alldata[allvars[varnum]][indx], 'o', 
            '''
            loglog(alldata['Nprod'][indx], alldata[allvars[varnum]][indx], 'o', 
                   label=allvars[varnum] + clbl)

            xlabel('$N_x N_y N_z$')
    leg = legend(loc=9, ncol=2, bbox_to_anchor=(0.5, -0.1))
    # leg.get_frame().set_alpha(0.5)
    savefig(run, bbox_inches='tight')

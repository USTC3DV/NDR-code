import os



"""GPU Number
"""
gpuname=0



"""KillingFusion
"""
# KillingFusion Frog
confname='kfusion_toy.conf'
casename='kfusion_frog'
os.system('python -u exp_runner.py --mode validate_pretrained --conf ./confs/{} --is_continue --case {} --gpu {}'.format(confname,casename,gpuname))

# KillingFusion Duck
confname='kfusion_toy.conf'
casename='kfusion_duck'
os.system('python -u exp_runner.py --mode validate_pretrained --conf ./confs/{} --is_continue --case {} --gpu {}'.format(confname,casename,gpuname))

# KillingFusion Duck
confname='kfusion_snoopy.conf'
casename='kfusion_snoopy'
os.system('python -u exp_runner.py --mode validate_pretrained --conf ./confs/{} --is_continue --case {} --gpu {}'.format(confname,casename,gpuname))

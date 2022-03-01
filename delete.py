import shutil

exams = [ 7, 8, 14, 26 ]
bases = [ 
    'rescale_max_',
    'rescale_min_',
    'vanilla_'
]

for exam in exams:
    for base in bases:
        base_path = 'data/' + base + str(exam) 
        out_path = base_path + '/out'
        shutil.rmtree(out_path, ignore_errors=True)
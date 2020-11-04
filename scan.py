import subprocess as sp

for case, out in [('bench_simple',             'simple.txt'),
                  ('bench_streams_st',         'streams-st.txt'),
                  ('bench_streams_mt',         'streams-mt.txt'),
                  ('bench_graph',              'graphs-single.txt'),
                  ('bench_graph_split',        'graphs-split.txt'),
                  ('bench_graph_update',       'graphs-update.txt'),
                  ('bench_graph_split_update', 'graphs-split-update.txt'),]:
    sp.run([f'make -B CASE={case}'], shell=True)
    sp.run([f'rm -f {out}'], shell=True)
    for epochs in [32]:
        for slots in [1, 2, 4, 8, 16]:
            for kernels in [1, 2, 4, 8, 16]:
                for size in [20]:
                    for threads in [128]:
                        for repetitions in [10]:
                            sp.run(f'srun bench {epochs} {slots} {kernels} {size} {threads} {repetitions} >> {out}', shell=True)
